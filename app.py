# ========================================
# AI 徒手 HIIT 巡迴健身教練 - Render 雲端版
# 架構：Browser getUserMedia -> POST frame -> Flask MediaPipe -> JSON
# ========================================
import base64
import os
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, jsonify, request
from groq import Groq


app = Flask(__name__)

mp_pose = None
pose = None
mp_draw = None

def get_pose_detector():
    global mp_pose, pose, mp_draw

    if pose is None:
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    return pose, mp_pose, mp_draw

# Render/Gunicorn 可能有多個請求同時進來，狀態更新加鎖避免競態
state_lock = threading.Lock()

# Groq API：請在 Render Environment Variables 設定 GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


CALORIE_METRICS = {
    'jumping_jacks': 0.15,
    'wall_sit': 0.35,
    'pushup': 0.45,
    'sit_up': 0.15,
    'mountain_climber': 0.15,
    'squat': 0.40,
    'triceps_dip': 0.30,
    'plank': 0.08,
    'high_knees': 0.18,
    'lunge': 0.38,
    'pushup_rotation': 0.55,
    'side_plank': 0.45
}

EXERCISE_TARGETS = {
    'beginner': {
        'jumping_jacks': 15,
        'wall_sit': 12,
        'pushup': 10,
        'sit_up': 15,
        'mountain_climber': 30,
        'squat': 15,
        'triceps_dip': 12,
        'plank': 30,
        'high_knees': 20,
        'lunge': 16,
        'pushup_rotation': 10,
        'side_plank': 30
    },
    'advanced': {
        'jumping_jacks': 30,
        'wall_sit': 24,
        'pushup': 20,
        'sit_up': 30,
        'mountain_climber': 60,
        'squat': 30,
        'triceps_dip': 20,
        'plank': 60,
        'high_knees': 40,
        'lunge': 30,
        'pushup_rotation': 20,
        'side_plank': 30
    }
}

exercise_state = {
    'is_started': False,
    'type': 'jumping_jacks',
    'level': 'beginner',
    'weight': 65.0,
    'target': EXERCISE_TARGETS['beginner']['jumping_jacks'],
    'counter': 0.0,
    'total_calories': 0.0,
    'stage': None,
    'hint': '請允許瀏覽器攝像頭權限，並開始訓練。',
    'start_time': None,
    'end_time': None,
    'last_tick': time.time(),
    'last_active': 0.0
}


def calculate_angle(a, b, c):
    """計算三點之間的角度，b 為頂點。"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def public_state():
    """只回傳前端需要的安全欄位。"""
    return {
        'counter': float(exercise_state.get('counter', 0.0)),
        'target': float(exercise_state.get('target', 0.0)),
        'hint': exercise_state.get('hint', ''),
        'total_calories': float(exercise_state.get('total_calories', 0.0)),
        'is_started': bool(exercise_state.get('is_started', False)),
        'type': exercise_state.get('type', ''),
        'level': exercise_state.get('level', '')
    }


def decode_base64_frame(data_url):
    """接收前端 canvas.toDataURL('image/jpeg')，轉成 OpenCV BGR 圖片。"""
    if not data_url:
        raise ValueError("frame is empty")

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cannot decode frame")
    return frame


def analyze_landmarks(landmarks):
    """保留原本 12 項運動判定核心邏輯。"""
    global exercise_state

    l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    l_kn = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    l_an = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    r_kn = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    r_an = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    vis_l_an = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
    vis_r_an = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility

    ex_type = exercise_state['type']
    weight_mul = exercise_state.get('weight', 65.0) / 65.0

    current_time = time.time()
    dt = current_time - exercise_state.get('last_tick', current_time)
    exercise_state['last_tick'] = current_time

    # ==========================================
    # 1. 開合跳
    if ex_type == 'jumping_jacks':
        arms_up = l_wr[1] < l_sh[1] and r_wr[1] < r_sh[1]
        legs_spread = abs(l_an[0] - r_an[0]) > 0.15
        if arms_up and legs_spread:
            if exercise_state['stage'] == 'down':
                exercise_state['counter'] += 1
                exercise_state['total_calories'] += CALORIE_METRICS['jumping_jacks'] * weight_mul
                exercise_state['hint'] = '漂亮！手腳打開到位，回到併攏後繼續下一下'
            else:
                exercise_state['hint'] = '手舉過肩、雙腳打開標準；下一步請收回手腳'
            exercise_state['stage'] = 'up'
        elif not arms_up and not legs_spread:
            exercise_state['stage'] = 'down'
            exercise_state['hint'] = '準備動作完成：雙手放下、雙腳併攏，接著向外跳開'
        else:
            exercise_state['hint'] = '開合跳要同時完成：雙手舉過肩，雙腳向外打開'

    # 2. 靠牆屈膝
    elif ex_type == 'wall_sit':
        knee_angle = calculate_angle(l_hip, l_kn, l_an)
        hip_angle = calculate_angle(l_sh, l_hip, l_kn)
        if knee_angle > 150 and hip_angle > 150:
            if exercise_state['stage'] == 'down':
                exercise_state['counter'] += 1
                exercise_state['total_calories'] += CALORIE_METRICS['wall_sit'] * weight_mul
                exercise_state['hint'] = '站起完成一次！下一次請慢慢坐到大腿接近水平'
            else:
                exercise_state['hint'] = '請背部靠穩牆面，慢慢下蹲到膝蓋約 90 度'
            exercise_state['stage'] = 'up'
        elif 70 < knee_angle < 110 and 70 < hip_angle < 110:
            exercise_state['stage'] = 'down'
            exercise_state['hint'] = '蹲姿到位！保持膝蓋約 90 度，再站起完成一次'
        else:
            exercise_state['hint'] = '調整姿勢：背靠牆、膝蓋彎曲到約 90 度，避免半蹲太高'

    # 3. 伏地挺身 (嚴格防弊)
    elif ex_type == 'pushup':  # 如果運動類型是伏地挺身
        l_angle = calculate_angle(l_sh, l_el, l_wr)  # 計算左臂肘部角度（肩-肘-腕）
        r_angle = calculate_angle(r_sh, r_el, r_wr)  # 計算右臂肘部角度（肩-肘-腕）
        avg_arm_angle = (l_angle + r_angle) / 2  # 計算雙臂肘部角度的平均值
        body_angle = calculate_angle(l_sh, l_hip, l_an)  # 計算身體的打直程度（肩-髖-踝）

        is_pushup_posture = (max(vis_l_an, vis_r_an) > 0.5) and (body_angle > 140) and (abs(l_sh[1] - l_hip[1]) < 0.35)  # 檢查是否呈現標準伏地挺身姿勢：踝部入鏡、身體打直、肩髖垂直

        if not is_pushup_posture:  # 如果姿勢不標準
            exercise_state['hint'] = '請確保全身入鏡，並呈現趴下撐體姿勢'  # 提示調整姿勢
        else:  # 如果姿勢標準
            if avg_arm_angle > 150:  # 如果雙臂平均角度 > 150（手臂伸直狀態）
                if exercise_state['stage'] == 'down':  # 如果之前是下壓狀態
                    exercise_state['counter'] += 1  # 計數加 1
                    exercise_state['total_calories'] += CALORIE_METRICS['pushup'] * weight_mul  # 增加熱量消耗
                    exercise_state['hint'] = '漂亮！手臂已撐直，完成一次伏地挺身'  # 確認完成一次
                else:  # 如果不是下壓狀態
                    exercise_state['hint'] = '身體保持一直線，接著彎曲手肘下壓'  # 提示下一步下壓
                exercise_state['stage'] = 'up'  # 標記為撐起狀態
            elif avg_arm_angle <= 90:  # 如果雙臂平均角度 <= 90（手肘彎曲狀態）
                exercise_state['stage'] = 'down'  # 標記為下壓狀態
                exercise_state['hint'] = '下壓到位！核心收緊，現在推起來'  # 提示推起
            else:  # 其他中間角度
                exercise_state['hint'] = '持續動作：下壓時手肘小於 90 度，撐起時手臂打直'  # 提示繼續動作

    # 4. 仰臥起坐 (防站立彎腰版)
    elif ex_type == 'sit_up':
        if exercise_state['stage'] is None:  # 如果是第一次開始仰臥起坐
            exercise_state['stage'] = 'up'  # 初始化狀態為「躺平」

        # 獲取肩、腰、膝蓋的能見度（檢查是否全身入鏡）
        vis_hip = max(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility)  # 取左右髖部能見度的最大值
        vis_kn = max(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility)  # 取左右膝蓋能見度的最大值
        vis_sh = max(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility)  # 取左右肩膀能見度的最大值

        if vis_hip < 0.4 or vis_kn < 0.4 or vis_sh < 0.4:  # 如果任何部位能見度低於 0.4
            exercise_state['hint'] = '調整位置：確保全身入鏡（肩、腰、膝蓋）'  # 提示調整位置
        else:  # 如果全身入鏡
            is_lying = abs(l_hip[1] - l_an[1]) < 0.35 and abs(r_hip[1] - r_an[1]) < 0.35  # 檢查髖部和踝部的垂直距離

            if not is_lying:  # 如果沒有躺平
                exercise_state['hint'] = '請躺平！不要站著彎腰'  # 提示躺平
            else:  # 如果躺平
                l_angle = calculate_angle(l_sh, l_hip, l_kn)  # 計算左側身體角度
                r_angle = calculate_angle(r_sh, r_hip, r_kn)  # 計算右側身體角度
                body_angle = min(l_angle, r_angle)  # 取最小角度

                dist_l = np.hypot(l_el[0] - l_kn[0], l_el[1] - l_kn[1])  # 計算左肘到膝蓋的距離
                dist_r = np.hypot(r_el[0] - r_kn[0], r_el[1] - r_kn[1])  # 計算右肘到膝蓋的距離
                min_dist = min(dist_l, dist_r)  # 取最小距離


                # 階段 A：起身到位，只標記為 up，不加次數
                if body_angle < 95 and min_dist < 0.22:
                    if exercise_state['stage'] == 'down':
                        exercise_state['stage'] = 'up'
                        exercise_state['hint'] = '起身到位！現在慢慢躺回地面，躺下後才會計數'
                    else:
                        exercise_state['hint'] = '已起身到位，請躺回地面完成這一下'

                # 階段 B：從 up 狀態躺回地面，才正式 +1
                elif body_angle > 120:
                    if exercise_state['stage'] == 'up':
                        exercise_state['counter'] += 1
                        exercise_state['total_calories'] += CALORIE_METRICS['sit_up'] * weight_mul
                        exercise_state['hint'] = '完成一次！下背貼地後，再準備下一次起身'
                    else:
                        exercise_state['hint'] = '下背貼地，準備起身'
                    exercise_state['stage'] = 'down'

                else:
                    exercise_state['hint'] = '繼續動作：起身靠近膝蓋後，再躺回地面才算一次'

    # 5. 登山者 (Mountain Climbers) - 雙腿獨立追蹤版
    elif ex_type == 'mountain_climber':  # 如果運動類型是登山者
        if not isinstance(exercise_state['stage'], dict):  # 若 stage 不是 dict，初始化為左右兩邊的狀態
            exercise_state['stage'] = {'l': 'down', 'r': 'down'}  # 設定左右初始為 down

        is_plank = abs(l_sh[1] - l_hip[1]) < 0.35 and abs(r_sh[1] - r_hip[1]) < 0.35  # 判斷是否為伏地挺身預備姿勢（肩與髖垂直）

        if not is_plank:  # 若不是伏地挺身姿勢
            exercise_state['hint'] = '請趴下呈現伏地挺身預備姿勢'  # 提示使用者趴下
        else:  # 已為伏地挺身姿勢則繼續判斷左右腿動作
            l_ang = calculate_angle(l_sh, l_hip, l_kn)  # 計算左腿髖-膝-踝角度
            r_ang = calculate_angle(r_sh, r_hip, r_kn)  # 計算右腿髖-膝-踝角度

            if l_ang < 100:  # 左腿抬起（角度變小代表往胸口靠近）
                if exercise_state['stage']['l'] == 'down':  # 如果左腿之前是 down，表示完成一次抬腿動作
                    exercise_state['counter'] += 1  # 計數增加 1
                    exercise_state['total_calories'] += CALORIE_METRICS.get('mountain_climber', 0.15) * weight_mul  # 增加熱量消耗
                    exercise_state['hint'] = '左腳漂亮！'  # 顯示即時鼓勵提示
                exercise_state['stage']['l'] = 'up'  # 將左腿狀態設為 up（抬起）
            elif l_ang > 120:  # 左腿收回到後方（角度變大）
                exercise_state['stage']['l'] = 'down'  # 將左腿狀態設為 down（收回）

            if r_ang < 100:  # 右腿抬起（角度變小代表往胸口靠近）
                if exercise_state['stage']['r'] == 'down':  # 如果右腿之前是 down，表示完成一次抬腿動作
                    exercise_state['counter'] += 1  # 計數增加 1
                    exercise_state['total_calories'] += CALORIE_METRICS.get('mountain_climber', 0.15) * weight_mul  # 增加熱量消耗
                    exercise_state['hint'] = '右腳漂亮！'  # 顯示即時鼓勵提示
                exercise_state['stage']['r'] = 'up'  # 將右腿狀態設為 up（抬起）
            elif r_ang > 120:  # 右腿收回到後方（角度變大）
                exercise_state['stage']['r'] = 'down'  # 將右腿狀態設為 down（收回）

            if l_ang > 120 and r_ang > 120:  # 兩腿都收回，提醒保持伏地挺身預備姿勢
                exercise_state['hint'] = '保持伏地挺身預備姿勢，左右膝輪流往胸口收'  # 顯示姿勢維持提示
            elif 100 <= l_ang <= 120 and 100 <= r_ang <= 120:  # 若兩腿角度處於中間區間
                exercise_state['hint'] = '膝蓋再往胸口靠近一點，角度不足還不會計數'  # 提示增加抬膝幅度

    # 6. 深蹲
    elif ex_type == 'squat':
        angle = calculate_angle(l_hip, l_kn, l_an)
        if angle > 160:
            if exercise_state['stage'] == 'down':
                exercise_state['counter'] += 1
                exercise_state['total_calories'] += CALORIE_METRICS['squat'] * weight_mul
                exercise_state['hint'] = '站直完成一次！下一次臀部往後坐、膝蓋穩定'
            else:
                exercise_state['hint'] = '準備深蹲：雙腳站穩，臀部往後往下坐'
            exercise_state['stage'] = 'up'
        elif angle < 90:
            exercise_state['stage'] = 'down'
            exercise_state['hint'] = '深蹲深度到位！保持胸口打開，現在站起來'
        else:
            exercise_state['hint'] = '再蹲低一點，膝蓋角度小於 90 度後再站起才會計數'

    # ==========================================
    # 7. 椅子三頭肌撐體 (Triceps Dip) - 垂直高度差防呆版
    # ==========================================
    elif ex_type == 'triceps_dip':
        # 1. 取得軀幹長度作為比例尺，避免因離鏡頭遠近造成誤差
        torso = max(abs(l_sh[1] - l_hip[1]), abs(r_sh[1] - r_hip[1]))
        if torso < 0.1: 
            torso = 0.2 

        # 2. 計算肩膀與手腕的高度差比例 (Y 軸往下為正，手腕 Y 減 肩膀 Y)
        # 撐起時：手腕在肩膀下方 (Y軸差距大，數值接近 1.0)
        # 降下時：肩膀降至接近手腕 (Y軸差距小，數值接近 0.5)
        l_dist = (l_wr[1] - l_sh[1]) / torso
        r_dist = (r_wr[1] - r_sh[1]) / torso
        avg_dist = (l_dist + r_dist) / 2

        # 3. 狀態機判定：用垂直距離取代容易失準的 2D 手肘角度
        if avg_dist < 0.5: 
            # 身體已降下，肩膀接近手腕
            if exercise_state['stage'] != 'down':
                exercise_state['stage'] = 'down'
                exercise_state['hint'] = '推起來！'
        elif avg_dist > 0.8: 
            # 手臂撐直，身體抬高
            if exercise_state['stage'] == 'down':
                exercise_state['counter'] += 1
                exercise_state['total_calories'] += CALORIE_METRICS['triceps_dip'] * weight_mul
                exercise_state['hint'] = '漂亮！手臂撐直完成一次，接著慢慢下壓'
            else:
                exercise_state['hint'] = '手掌撐穩椅緣，身體靠近椅子，準備往下沉'
            exercise_state['stage'] = 'up'
        else:
            exercise_state['hint'] = '繼續控制身體：下降到肩膀接近手腕，再用三頭肌推起'

    # 8. 平板支撐 (計時修正版)
    elif ex_type == 'plank':
        body_angle = calculate_angle(l_sh, l_hip, l_an)
        if 160 < body_angle <= 180:
            exercise_state['counter'] += dt
            exercise_state['total_calories'] += CALORIE_METRICS['plank'] * dt * weight_mul
            time_left = max(0, int(exercise_state['target'] - exercise_state['counter']))
            exercise_state['hint'] = f'姿勢標準，撐住！剩下 {time_left} 秒'
        else:
            exercise_state['hint'] = '臀部請壓低！計時暫停中'

    # ★ 9. 原地抬膝 (已明確補上姿勢指導文字，排除計時干擾)
    elif ex_type == 'high_knees':
        l_up = l_kn[1] < l_hip[1] + 0.05
        r_up = r_kn[1] < r_hip[1] + 0.05

        if l_up or r_up:
            if exercise_state['stage'] == 'down':
                exercise_state['counter'] += 1
                exercise_state['total_calories'] += CALORIE_METRICS['high_knees'] * weight_mul
                exercise_state['hint'] = '做得好！膝蓋盡量抬高至腰部高度'
            exercise_state['stage'] = 'up'
        elif not l_up and not r_up:
            exercise_state['stage'] = 'down'
            # 靜止或腳放下時的純姿勢建議
            exercise_state['hint'] = '背部打直，核心收緊，左右腳快速交替高抬'

    # 10. 弓步深蹲
    elif ex_type == 'lunge':
        l_ang = calculate_angle(l_hip, l_kn, l_an)
        r_ang = calculate_angle(r_hip, r_kn, r_an)
        if l_ang < 110 and r_ang < 110:
            if exercise_state['stage'] == 'up':
                exercise_state['counter'] += 1
                exercise_state['total_calories'] += CALORIE_METRICS['lunge'] * weight_mul
                exercise_state['hint'] = '弓步深度到位，完成一次！推回站姿後換腳'
            else:
                exercise_state['hint'] = '下蹲到位！前後腳膝蓋都彎曲，保持身體穩定'
            exercise_state['stage'] = 'down'
        elif l_ang > 150 or r_ang > 150:
            exercise_state['stage'] = 'up'
            exercise_state['hint'] = '站姿準備完成：向前或向後跨一步，身體垂直下沉'
        else:
            exercise_state['hint'] = '弓步再下沉一點，前後膝都接近彎曲後才會計數'

    # ==========================================
    # 11. 掌上壓後轉身 (Pushup Rotation) - 順序防呆版
    # ==========================================
    elif ex_type == 'pushup_rotation':
        # 1. 拔除容易誤判的腳踝能見度，改看身體打直角度即可 (大幅提升容錯率)
        body_angle = calculate_angle(l_sh, l_hip, l_kn) # 改看膝蓋，避免腳踝在畫面邊緣被裁切

        # 如果身體彎曲太嚴重 (例如站著彎腰)，才提示姿勢不佳
        if body_angle < 120:
            exercise_state['hint'] = '請保持背部與腿部呈一直線'
        else:
            # 2. 取得雙手肘部角度
            l_angle = calculate_angle(l_sh, l_el, l_wr)
            r_angle = calculate_angle(r_sh, r_el, r_wr)
            avg_arm_angle = (l_angle + r_angle) / 2

            # 3. 判斷轉身舉手 (有一隻手腕的 Y 軸舉得比肩膀高出許多)
            is_rot = (l_wr[1] < l_sh[1] - 0.15) or (r_wr[1] < r_sh[1] - 0.15)

            # 4. 嚴格順序狀態機：必須先「下壓」才能「舉手」
            # 階段 A：確實完成伏地挺身下壓動作 (雙手平均彎曲小於 90 度)
            if avg_arm_angle < 90:
                exercise_state['stage'] = 'down'
                exercise_state['hint'] = '推起並轉身舉手！'

            # 階段 B：從下壓狀態撐起，並且舉手轉身
            elif exercise_state['stage'] == 'down':
                # 當偵測到轉身舉手時，必須保證「另一隻撐在地上的手是伸直的(>140度)」
                # 這可以完美防止「躺在地上直接舉手」或「站著亂揮手」的作弊
                if is_rot and (l_angle > 140 or r_angle > 140):
                    exercise_state['counter'] += 1
                    exercise_state['total_calories'] += CALORIE_METRICS['pushup_rotation'] * weight_mul
                    exercise_state['hint'] = '完美轉身！換邊繼續'
                    exercise_state['stage'] = 'up' # 結算後重置，等待下一次的下壓

    # ==========================================
    # 12. 側平板式 (Side Plank) - 靜態維持防偷懶版
    # ==========================================
    elif ex_type == 'side_plank':
        # 1. 初始化方向狀態 ('left_side' 代表左肘撐地右手舉高，'right_side' 反之)
        if exercise_state['stage'] not in ['left_side', 'right_side']:
            exercise_state['stage'] = 'left_side'

        # 2. 核心骨架角度計算
        # 計算兩邊 脖子(肩中點)-髖部-膝蓋 的打直程度，確保身體沒有掉到地上
        l_hip_line = calculate_angle(l_sh, l_hip, l_kn)
        r_hip_line = calculate_angle(r_sh, r_hip, r_kn)

        # 3. 偵測目前是哪一邊在撐地與舉手
        is_holding = False
        current_side = exercise_state['stage']

        if current_side == 'left_side':
            # 左肘撐地狀況：身體右側朝上
            # 條件 A：右手腕(r_wr)必須明顯高於右肩膀(r_sh)，代表右手高舉
            # 條件 B：右側身體必須打直 (角度 > 150)，代表屁股有撐起來，沒有躺下
            r_arm_up = r_wr[1] < r_sh[1] - 0.15
            r_body_straight = r_hip_line > 150

            if r_arm_up and r_body_straight:
                is_holding = True

        elif current_side == 'right_side':
            # 右肘撐地狀況：身體左側朝上
            # 條件 A：左手腕(l_wr)必須明顯高於左肩膀(l_sh)
            # 條件 B：左側身體必須打直 (角度 > 150)
            l_arm_up = l_wr[1] < l_sh[1] - 0.15
            l_body_straight = l_hip_line > 150

            if l_arm_up and l_body_straight:
                is_holding = True

        # 4. 狀態機與動態秒數結算
        if is_holding:
            # 姿勢完全標準，開始扣秒數！
            exercise_state['counter'] += dt
            exercise_state['total_calories'] += CALORIE_METRICS['side_plank'] * dt * weight_mul

            # 計算當前側邊撐了幾秒
            current_side_seconds = exercise_state['counter']

            if current_side == 'left_side':
                if current_side_seconds < 15.0:
                    time_left = max(0, int(15.0 - current_side_seconds))
                    exercise_state['hint'] = f'🟢 左肘撐地中！保持核心收緊，右手舉高，剩 {time_left} 秒'
                else:
                    # 左邊滿 15 秒，自動切換到右邊
                    exercise_state['stage'] = 'right_side'
                    exercise_state['hint'] = '⏱️ 時間到！請立刻翻身換右肘撐地，左手舉高！'

            elif current_side == 'right_side':
                # 右邊的計時是從 15 秒一路上升到 30 秒
                if current_side_seconds < 30.0:
                    time_left = max(0, int(30.0 - current_side_seconds))
                    exercise_state['hint'] = f'🟢 右肘撐地中！最後衝刺，左手舉高，剩 {time_left} 秒'
                else:
                    exercise_state['hint'] = '🎉 太棒了！雙側側平板皆挑戰成功！'
        else:
            # 沒撐住、手放下或躺下了，計時自動暫停
            if current_side == 'left_side':
                exercise_state['hint'] = '⚠️ 沒撐住或手放下了喔！請左肘撐起、右手舉高，計時暫停中'
            else:
                exercise_state['hint'] = '⚠️ 右邊快完成了，加油！請右肘撐起、左手舉高，計時暫停中'


    return public_state()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok'})


@app.route('/start_exercise', methods=['POST'])
def start_exercise_api():
    data = request.get_json(silent=True) or {}
    ex_type = data.get('type', 'jumping_jacks')
    level = data.get('level', 'beginner')

    if level not in EXERCISE_TARGETS:
        return jsonify({'status': 'error', 'message': 'invalid level'}), 400
    if ex_type not in EXERCISE_TARGETS[level]:
        return jsonify({'status': 'error', 'message': 'invalid exercise type'}), 400

    try:
        weight = float(data.get('weight', 65.0))
    except (TypeError, ValueError):
        weight = 65.0

    with state_lock:
        exercise_state['type'] = ex_type
        exercise_state['level'] = level
        exercise_state['weight'] = weight
        exercise_state['target'] = EXERCISE_TARGETS[level][ex_type]
        exercise_state['counter'] = 0.0
        exercise_state['stage'] = None
        exercise_state['hint'] = '偵測已開始，請進入鏡頭範圍。'
        exercise_state['last_tick'] = time.time()
        exercise_state['last_active'] = time.time()
        exercise_state['is_started'] = True

        # 只在巡迴第一項開始時重置累計熱量
        if exercise_state['start_time'] is None:
            exercise_state['start_time'] = time.time()
            exercise_state['total_calories'] = 0.0
            exercise_state['end_time'] = None

        state = public_state()

    return jsonify({'status': 'success', **state})


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame_api():
    data = request.get_json(silent=True) or {}

    with state_lock:
        if not exercise_state.get('is_started'):
            return jsonify(public_state())

    try:
        frame = decode_base64_frame(data.get('frame'))
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            pose_detector, mp_pose, mp_draw = get_pose_detector()
            results = pose_detector.process(img_rgb)

        with state_lock:
            if not exercise_state.get('is_started'):
                return jsonify(public_state())

            if results.pose_landmarks:
                analyze_landmarks(results.pose_landmarks.landmark)
            else:
                exercise_state['hint'] = '未偵測到完整人體，請退後一點並保持全身入鏡。'

            return jsonify(public_state())

    except Exception as e:
        with state_lock:
            exercise_state['hint'] = f'影像分析失敗：{str(e)}'
            return jsonify(public_state()), 400


@app.route('/get_stats')
def get_stats():
    with state_lock:
        return jsonify(public_state())


@app.route('/stop_exercise', methods=['POST'])
def stop_exercise_api():
    with state_lock:
        exercise_state['is_started'] = False
        exercise_state['end_time'] = time.time()
        return jsonify({'status': 'success', **public_state()})


@app.route('/pause_exercise', methods=['POST'])
def pause_exercise_api():
    with state_lock:
        exercise_state['is_started'] = False
        exercise_state['hint'] = '已暫停偵測，重看教學結束後會從目前進度繼續。'
        return jsonify({'status': 'success', **public_state()})


@app.route('/resume_exercise', methods=['POST'])
def resume_exercise_api():
    with state_lock:
        exercise_state['is_started'] = True
        exercise_state['last_tick'] = time.time()
        exercise_state['last_active'] = time.time()
        exercise_state['hint'] = '已恢復偵測，請回到鏡頭前繼續。'
        return jsonify({'status': 'success', **public_state()})


@app.route('/reset_circuit', methods=['POST'])
def reset_circuit_api():
    with state_lock:
        exercise_state.update({
            'is_started': False,
            'type': 'jumping_jacks',
            'level': 'beginner',
            'weight': 65.0,
            'target': EXERCISE_TARGETS['beginner']['jumping_jacks'],
            'counter': 0.0,
            'total_calories': 0.0,
            'stage': None,
            'hint': '請允許瀏覽器攝像頭權限，並開始訓練。',
            'start_time': None,
            'end_time': None,
            'last_tick': time.time(),
            'last_active': 0.0
        })
        return jsonify({'status': 'success', **public_state()})


@app.route('/get_ai_feedback', methods=['POST'])
def get_ai_feedback():
    with state_lock:
        start_time = exercise_state.get('start_time')
        end_time = exercise_state.get('end_time') or time.time()
        total_calories = float(exercise_state.get('total_calories', 0.0))
        level = exercise_state.get('level', 'beginner')

    if not start_time:
        duration = 0
    else:
        duration = max(0, end_time - start_time)

    mins, secs = int(duration // 60), int(duration % 60)
    level_text = "進階" if level == "advanced" else "初階"

    prompt = f"""你現在是專業居家徒手 HIIT 健身教練。學員完成了 12 項巡迴訓練。
總計時間：{mins}分{secs}秒。總消耗：{total_calories:.2f}大卡。
訓練強度：{level_text}。
請用繁體中文，簡潔鼓勵學員，針對徒手訓練給予恢復、補水與下次訓練建議。絕對禁止提到啞鈴或任何負重器材。"""

    if client is None:
        fallback = (
            f"已完成本次徒手 HIIT 巡迴。總時間約 {mins} 分 {secs} 秒，"
            f"累計消耗 {total_calories:.2f} kcal。\n\n"
            "提醒：Render 尚未設定 GROQ_API_KEY，所以目前使用本機備用講評。"
            "請補充水分、做 5 到 10 分鐘伸展，下一次可優先改善提示中最常出現的動作問題。"
        )
        return jsonify({"feedback": fallback})

    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        feedback = chat.choices[0].message.content
        with state_lock:
            exercise_state['start_time'] = None
        return jsonify({"feedback": feedback})
    except Exception as e:
        return jsonify({"feedback": f"AI 分析失敗：{str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
