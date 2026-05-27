# ========================================
# AI 徒手 HIIT 巡迴健身教練 - Render 雲端版
# 架構：Browser getUserMedia -> POST frame -> Flask MediaPipe -> JSON
# ========================================

# 基礎函式庫引入
import base64          # 用於解碼前端傳來的 Base64 編碼影像數據
import os              # 用於讀取環境變數（如 GROQ_API_KEY）
import time            # 用於計時和時間戳記
import threading       # 用於執行緒鎖，保護多個請求同時訪問狀態的競態問題
import uuid            # 用於生成唯一的 client_id，識別不同的裝置用戶

# 電腦視覺與影像處理
import cv2             # OpenCV，用於影像解碼和格式轉換
import mediapipe as mp # MediaPipe，用於人體姿勢偵測
import numpy as np     # NumPy，用於數學計算（角度計算）

# Web 框架與 API 呼叫
from flask import Flask, render_template, jsonify, request  # Flask Web 框架用於建立 API 端點
from groq import Groq  # Groq API 客戶端，用於 AI 教練評論生成


app = Flask(__name__)  # 初始化 Flask 應用程式

# 全域 MediaPipe 姿勢偵測器變數（避免重複初始化）
mp_pose = None         # MediaPipe Pose 模組
pose = None            # MediaPipe Pose 偵測器實例
mp_draw = None         # MediaPipe 繪圖工具（未來可用於畫骨骼節點）

def get_pose_detector():
    # 函式：延遲初始化 MediaPipe Pose 偵測器（第一次調用時初始化，之後重複使用）
    global mp_pose, pose, mp_draw

    if pose is None:  # 如果偵測器還未初始化
        mp_pose = mp.solutions.pose  # 載入 MediaPipe Pose 模組
        mp_draw = mp.solutions.drawing_utils  # 載入繪圖工具（備用）
        pose = mp_pose.Pose(  # 建立 Pose 偵測器實例
            static_image_mode=False,  # 影片模式（即時追蹤）
            model_complexity=0,  # 模型複雜度 0（最輕量，最快）
            enable_segmentation=False,  # 不進行身體分割（節省資源）
            min_detection_confidence=0.5,  # 最小偵測信心度 50%
            min_tracking_confidence=0.5  # 最小追蹤信心度 50%
        )

    return pose, mp_pose, mp_draw  # 回傳偵測器、模組和繪圖工具


# 執行緒安全的狀態鎖（防止多個請求同時修改狀態）
state_lock = threading.Lock()

# Groq API 配置：從環境變數讀取 API 金鑰
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # 讀取環境變數中的 GROQ_API_KEY
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None  # 如果有金鑰則建立客戶端，否則為 None（離線模式）


# 運動類型對應的卡路里消耗係數（單位：大卡/秒）
CALORIE_METRICS = {
    'jumping_jacks': 0.15,  # 開合跳：每秒消耗 0.15 大卡
    'wall_sit': 0.35,  # 靠牆屈膝：每秒消耗 0.35 大卡
    'pushup': 0.45,  # 伏地挺身：每秒消耗 0.45 大卡
    'sit_up': 0.15,  # 仰臥起坐：每秒消耗 0.15 大卡
    'mountain_climber': 0.15,  # 登山者：每秒消耗 0.15 大卡
    'squat': 0.40,  # 深蹲：每秒消耗 0.40 大卡
    'triceps_dip': 0.30,  # 三頭肌撐體：每秒消耗 0.30 大卡
    'high_knees': 0.18,  # 原地抬膝：每秒消耗 0.18 大卡
    'lunge': 0.38,  # 弓步深蹲：每秒消耗 0.38 大卡
    'pushup_rotation': 0.55  # 掌上壓後轉身：每秒消耗 0.55 大卡
}


# 各個訓練強度下，每項運動的目標次數
EXERCISE_TARGETS = {
    'beginner': {  # 初階目標次數
        'jumping_jacks': 15,  # 開合跳：15 次
        'wall_sit': 12,  # 靠牆屈膝：12 次
        'pushup': 10,  # 伏地挺身：10 次
        'sit_up': 15,  # 仰臥起坐：15 次
        'mountain_climber': 30,  # 登山者：30 次（單腳）
        'squat': 15,  # 深蹲：15 次
        'triceps_dip': 12,  # 三頭肌撐體：12 次
        'high_knees': 20,  # 原地抬膝：20 次
        'lunge': 16,  # 弓步深蹲：16 次（單邊）
        'pushup_rotation': 10  # 掌上壓後轉身：10 次
    },
    'advanced': {  # 進階目標次數
        'jumping_jacks': 30,  # 開合跳：30 次
        'wall_sit': 24,  # 靠牆屈膝：24 次
        'pushup': 20,  # 伏地挺身：20 次
        'sit_up': 30,  # 仰臥起坐：30 次
        'mountain_climber': 60,  # 登山者：60 次（單腳）
        'squat': 30,  # 深蹲：30 次
        'triceps_dip': 20,  # 三頭肌撐體：20 次
        'high_knees': 40,  # 原地抬膝：40 次
        'lunge': 30,  # 弓步深蹲：30 次（單邊）
        'pushup_rotation': 20  # 掌上壓後轉身：20 次
    }
}

def create_default_state():
    # 函式：為新用戶建立預設狀態
    return {
        'is_started': False,  # 訓練是否已開始
        'type': 'jumping_jacks',  # 當前運動類型（預設開合跳）
        'level': 'beginner',  # 訓練強度（初階/進階）
        'weight': 65.0,  # 體重（kg，預設 65 公斤用於卡路里計算）
        'target': EXERCISE_TARGETS['beginner']['jumping_jacks'],  # 目標次數
        'counter': 0.0,  # 已完成次數（浮點數）
        'total_calories': 0.0,  # 累計消耗卡路里
        'stage': None,  # 運動狀態機（如 'up', 'down' 等，用於判斷姿勢變化）
        'hint': '請允許瀏覽器攝像頭權限，並開始訓練。',  # 給用戶的提示文字
        'start_time': None,  # 巡迴訓練開始時間
        'end_time': None,  # 巡迴訓練結束時間
        'last_tick': time.time(),  # 上一次 tick 的時間戳記（用於計算時間差）
        'last_active': 0.0  # 上一次活動的時間戳記
    }


# 全域字典：儲存每台裝置的訓練狀態（key=client_id, value=exercise_state）
exercise_states = {}


def calculate_angle(a, b, c):
    # 函式：計算三點之間的角度（b 為頂點）
    # 使用向量叉積和反三角函數計算兩條邊的夾角
    a = np.array(a)  # 將第一點轉換為 NumPy 陣列
    b = np.array(b)  # 將頂點轉換為 NumPy 陣列
    c = np.array(c)  # 將第三點轉換為 NumPy 陣列
    
    # 計算兩條向量的反正切差異，轉換為角度（0-180 度）
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)  # 將弧度轉為度數，取絕對值
    if angle > 180.0:  # 如果角度大於 180 度
        angle = 360 - angle  # 取補角（保證結果在 0-180 之間）
    return angle  # 回傳角度值


def get_client_id():
    # 函式：從請求 header 獲取用戶端 ID，若無則生成新的 UUID
    client_id = request.headers.get("X-Client-ID")  # 從 HTTP header 讀取 X-Client-ID
    if not client_id:  # 如果 header 中沒有 client_id
        client_id = str(uuid.uuid4())  # 生成新的 UUID 作為 client_id
    return client_id  # 回傳 client_id


def get_user_state():
    # 函式：依照 client_id 取得該裝置自己的運動狀態（若無則新建）
    client_id = get_client_id()  # 取得 client_id
    if client_id not in exercise_states:  # 如果該 client_id 還沒有狀態
        exercise_states[client_id] = create_default_state()  # 為該 client 建立預設狀態
    return exercise_states[client_id]  # 回傳 client 的運動狀態


def public_state(state, landmarks=None):
    """只回傳前端需要的安全欄位。"""
    data = {
        'counter': float(state.get('counter', 0.0)),
        'target': float(state.get('target', 0.0)),
        'hint': state.get('hint', ''),
        'total_calories': float(state.get('total_calories', 0.0)),
        'is_started': bool(state.get('is_started', False)),
        'type': state.get('type', ''),
        'level': state.get('level', '')
    }

    if landmarks is not None:
        data['landmarks'] = [
            {
                'x': float(lm.x),
                'y': float(lm.y),
                'visibility': float(lm.visibility)
            }
            for lm in landmarks
        ]

    return data


def decode_base64_frame(data_url):
    # 函式：解碼前端傳來的 Base64 編碼影像，轉為 OpenCV BGR 圖片
    if not data_url:  # 如果沒有提供影像資料
        raise ValueError("frame is empty")  # 拋出錯誤

    if "," in data_url:  # 檢查是否為標準 data URL 格式（含 'data:image/jpeg;base64,' 前綴）
        _, encoded = data_url.split(",", 1)  # 分離 prefix 和 base64 編碼部分
    else:
        encoded = data_url  # 如果沒有 prefix，直接使用全部內容

    img_bytes = base64.b64decode(encoded)  # 解碼 base64 為二進位資料
    np_arr = np.frombuffer(img_bytes, np.uint8)  # 轉換為 NumPy 位元組陣列
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 用 OpenCV 解碼為 BGR 圖片
    if frame is None:  # 如果解碼失敗
        raise ValueError("cannot decode frame")  # 拋出錯誤
    return frame  # 回傳解碼後的影像


def analyze_landmarks(landmarks, state):
    # 函式：根據 MediaPipe 偵測的身體關節點，判定當前運動是否完成
    # 保留原本 12 項運動判定核心邏輯
    global mp_pose  # 使用全域 MediaPipe Pose 模組

    if mp_pose is None:  # 如果 MediaPipe 還未初始化
        get_pose_detector()  # 進行延遲初始化

    # 提取左側身體各關節的座標（x, y）
    l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  # 左肩
    l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]  # 左肘
    l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]  # 左腕
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  # 左髖
    l_kn = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]  # 左膝
    l_an = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]  # 左踝

    # 提取右側身體各關節的座標（x, y）
    r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]  # 右肩
    r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]  # 右肘
    r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]  # 右腕
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]  # 右髖
    r_kn = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]  # 右膝
    r_an = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]  # 右踝

    # 提取踝部的可見度（0-1，用於判斷是否入鏡）
    vis_l_an = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility  # 左踝能見度
    vis_r_an = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility  # 右踝能見度

    # 獲取運動類型和體重信息
    ex_type = state['type']  # 當前運動類型
    weight_mul = state.get('weight', 65.0) / 65.0  # 體重倍數（用於調整卡路里消耗）

    # 計算時間差（用於後續計時功能）
    current_time = time.time()  # 當前時間
    dt = current_time - state.get('last_tick', current_time)  # 時間差（秒）
    state['last_tick'] = current_time  # 更新上一次時間戳

    # ==========================================
    # 1. 開合跳 (Jumping Jacks) 判定邏輯
    # ==========================================
    if ex_type == 'jumping_jacks':  # 如果運動類型是開合跳
        # 判定手臂是否舉起（手腕 Y 座標小於肩膀 Y 座標表示抬起）
        arms_up = l_wr[1] < l_sh[1] and r_wr[1] < r_sh[1]  # 雙手都舉起時為 True
        # 判定雙腳是否打開（左右踝之間的水平距離超過 0.15 表示打開）
        legs_spread = abs(l_an[0] - r_an[0]) > 0.15  # 雙腳打開時為 True
        
        if arms_up and legs_spread:  # 如果手臂舉起且雙腳打開（完成動作）
            if state['stage'] == 'down':  # 如果之前是併攏狀態
                state['counter'] += 1  # 計數加 1
                state['total_calories'] += CALORIE_METRICS['jumping_jacks'] * weight_mul  # 增加卡路里消耗
                state['hint'] = '漂亮！手腳打開到位，回到併攏後繼續下一下'  # 提示下一步
            else:  # 如果不是併攏狀態（已經是抬起狀態）
                state['hint'] = '手舉過肩、雙腳打開標準；下一步請收回手腳'  # 提示準備收回
            state['stage'] = 'up'  # 標記為抬起狀態
            
        elif not arms_up and not legs_spread:  # 如果手臂放下且雙腳併攏（回到起始狀態）
            state['stage'] = 'down'  # 標記為併攏狀態
            state['hint'] = '準備動作完成：雙手放下、雙腳併攏，接著向外跳開'  # 提示下一步跳開
            
        else:  # 其他中間狀態（手和腳的動作不同步）
            state['hint'] = '開合跳要同時完成：雙手舉過肩，雙腳向外打開'  # 提示動作要同步


    # ==========================================
    # 2. 靠牆屈膝 (Wall Sit) 判定邏輯
    # ==========================================
    elif ex_type == 'wall_sit':  # 如果運動類型是靠牆屈膝
        # 計算膝蓋的彎曲角度（髖-膝-踝）
        knee_angle = calculate_angle(l_hip, l_kn, l_an)  # 左側膝蓋角度
        # 計算髖部的彎曲角度（肩-髖-膝）
        hip_angle = calculate_angle(l_sh, l_hip, l_kn)  # 左側髖部角度
        
        if knee_angle > 150 and hip_angle > 150:  # 如果膝蓋和髖都打直（> 150 度，接近直線）
            if state['stage'] == 'down':  # 如果之前是蹲下狀態
                state['counter'] += 1  # 計數加 1
                state['total_calories'] += CALORIE_METRICS['wall_sit'] * weight_mul  # 增加卡路里消耗
                state['hint'] = '站起完成一次！下一次請慢慢坐到大腿接近水平'  # 提示下一步
            else:  # 如果不是蹲下狀態
                state['hint'] = '請背部靠穩牆面，慢慢下蹲到膝蓋約 90 度'  # 提示蹲下
            state['stage'] = 'up'  # 標記為站起狀態
            
        elif 70 < knee_angle < 110 and 70 < hip_angle < 110:  # 如果膝蓋和髖都彎曲到 70-110 度（約 90 度）
            state['stage'] = 'down'  # 標記為蹲下狀態
            state['hint'] = '蹲姿到位！保持膝蓋約 90 度，再站起完成一次'  # 提示站起
            
        else:  # 其他中間狀態（姿勢不標準）
            state['hint'] = '調整姿勢：背靠牆、膝蓋彎曲到約 90 度，避免半蹲太高'  # 提示調整


    # ==========================================
    # 3. 伏地挺身 (Pushup) 判定邏輯 - 嚴格防弊
    # ==========================================
    elif ex_type == 'pushup':  # 如果運動類型是伏地挺身
        # 計算雙臂肘部的彎曲角度（肩-肘-腕）
        l_angle = calculate_angle(l_sh, l_el, l_wr)  # 左臂肘部角度
        r_angle = calculate_angle(r_sh, r_el, r_wr)  # 右臂肘部角度
        avg_arm_angle = (l_angle + r_angle) / 2  # 雙臂平均肘部角度
        # 計算身體的打直程度（肩-髖-踝）
        body_angle = calculate_angle(l_sh, l_hip, l_an)  # 身體角度
        
        # 檢查是否呈現標準伏地挺身姿勢（踝部入鏡、身體打直、肩髖垂直）
        is_pushup_posture = (max(vis_l_an, vis_r_an) > 0.5) and (body_angle > 140) and (abs(l_sh[1] - l_hip[1]) < 0.35)
        
        if not is_pushup_posture:  # 如果姿勢不標準
            state['hint'] = '請確保全身入鏡，並呈現趴下撐體姿勢'  # 提示調整姿勢
        else:  # 如果姿勢標準
            if avg_arm_angle > 150:  # 如果雙臂平均角度 > 150（手臂伸直狀態）
                if state['stage'] == 'down':  # 如果之前是下壓狀態
                    state['counter'] += 1  # 計數加 1
                    state['total_calories'] += CALORIE_METRICS['pushup'] * weight_mul  # 增加卡路里消耗
                    state['hint'] = '漂亮！手臂已撐直，完成一次伏地挺身'  # 確認完成一次
                else:  # 如果不是下壓狀態
                    state['hint'] = '身體保持一直線，接著彎曲手肘下壓'  # 提示下一步下壓
                state['stage'] = 'up'  # 標記為撐起狀態
                
            elif avg_arm_angle <= 90:  # 如果雙臂平均角度 <= 90（手肘彎曲狀態）
                state['stage'] = 'down'  # 標記為下壓狀態
                state['hint'] = '下壓到位！核心收緊，現在推起來'  # 提示推起
                
            else:  # 其他中間角度
                state['hint'] = '持續動作：下壓時手肘小於 90 度，撐起時手臂打直'  # 提示繼續動作

    # ==========================================
    # 4. 仰臥起坐 (Sit-up) 判定邏輯 - 防站立彎腰版
    # ==========================================
    elif ex_type == 'sit_up':  # 如果運動類型是仰臥起坐
        if state['stage'] is None:  # 如果是第一次開始仰臥起坐
            state['stage'] = 'up'  # 初始化狀態為「躺平」

        # 獲取各部位的能見度（檢查是否全身入鏡）
        vis_hip = max(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility)  # 髖部能見度
        vis_kn = max(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility)  # 膝蓋能見度
        vis_sh = max(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility)  # 肩膀能見度

        if vis_hip < 0.4 or vis_kn < 0.4 or vis_sh < 0.4:  # 如果任何部位能見度低於 0.4
            state['hint'] = '調整位置：確保全身入鏡（肩、腰、膝蓋）'  # 提示調整位置
        else:  # 如果全身入鏡
            # 檢查是否躺平（髖部和踝部的垂直距離小於 0.35）
            is_lying = abs(l_hip[1] - l_an[1]) < 0.35 and abs(r_hip[1] - r_an[1]) < 0.35

            if not is_lying:  # 如果沒有躺平
                state['hint'] = '請躺平！不要站著彎腰'  # 提示躺平
            else:  # 如果躺平
                # 計算身體彎曲角度（肩-髖-膝）
                l_angle = calculate_angle(l_sh, l_hip, l_kn)  # 左側身體角度
                r_angle = calculate_angle(r_sh, r_hip, r_kn)  # 右側身體角度
                body_angle = min(l_angle, r_angle)  # 取最小角度

                # 計算肘部與膝蓋的距離（判斷是否靠近膝蓋）
                dist_l = np.hypot(l_el[0] - l_kn[0], l_el[1] - l_kn[1])  # 左肘到膝蓋距離
                dist_r = np.hypot(r_el[0] - r_kn[0], r_el[1] - r_kn[1])  # 右肘到膝蓋距離
                min_dist = min(dist_l, dist_r)  # 取最小距離

                # 階段 A：起身到位，只標記為 up，不加次數
                if body_angle < 95 and min_dist < 0.22:  # 如果身體角度小於 95 度且肘靠近膝蓋
                    if state['stage'] == 'down':  # 如果之前是躺平狀態
                        state['stage'] = 'up'  # 標記為起身狀態
                        state['hint'] = '起身到位！現在慢慢躺回地面，躺下後才會計數'  # 提示躺下
                    else:  # 如果已經是起身狀態
                        state['hint'] = '已起身到位，請躺回地面完成這一下'  # 提示躺下

                # 階段 B：從 up 狀態躺回地面，才正式計數 +1
                elif body_angle > 120:  # 如果身體角度大於 120 度（接近躺平）
                    if state['stage'] == 'up':  # 如果之前是起身狀態
                        state['counter'] += 1  # 計數加 1
                        state['total_calories'] += CALORIE_METRICS['sit_up'] * weight_mul  # 增加卡路里消耗
                        state['hint'] = '完成一次！下背貼地後，再準備下一次起身'  # 提示下一步
                    else:  # 如果不是起身狀態
                        state['hint'] = '下背貼地，準備起身'  # 提示起身
                    state['stage'] = 'down'  # 標記為躺平狀態

                else:  # 其他中間狀態
                    state['hint'] = '繼續動作：起身靠近膝蓋後，再躺回地面才算一次'  # 提示繼續動作

    # ==========================================
    # 5. 登山者 (Mountain Climber) 判定邏輯 - 雙腿獨立追蹤版
    # ==========================================
    elif ex_type == 'mountain_climber':  # 如果運動類型是登山者
        # 初始化狀態為 dict，獨立追蹤左右腿
        if not isinstance(state['stage'], dict):  # 若 stage 不是 dict
            state['stage'] = {'l': 'down', 'r': 'down'}  # 初始化為左右都下放狀態

        # 檢查是否為伏地挺身預備姿勢（肩與髖垂直）
        is_plank = abs(l_sh[1] - l_hip[1]) < 0.35 and abs(r_sh[1] - r_hip[1]) < 0.35

        if not is_plank:  # 若不是伏地挺身姿勢
            state['hint'] = '請趴下呈現伏地挺身預備姿勢'  # 提示使用者趴下
        else:  # 已為伏地挺身姿勢則繼續判斷左右腿動作
            # 計算左右腿的髖-膝-踝角度
            l_ang = calculate_angle(l_sh, l_hip, l_kn)  # 左腿髖-膝-踝角度
            r_ang = calculate_angle(r_sh, r_hip, r_kn)  # 右腿髖-膝-踝角度

            # 判定左腿是否抬起
            if l_ang < 100:  # 左腿抬起（角度變小代表往胸口靠近）
                if state['stage']['l'] == 'down':  # 如果左腿之前是 down
                    state['counter'] += 1  # 計數增加 1
                    state['total_calories'] += CALORIE_METRICS.get('mountain_climber', 0.15) * weight_mul  # 增加卡路里消耗
                    state['hint'] = '左腳漂亮！'  # 顯示即時鼓勵提示
                state['stage']['l'] = 'up'  # 將左腿狀態設為 up
            elif l_ang > 120:  # 左腿收回到後方（角度變大）
                state['stage']['l'] = 'down'  # 將左腿狀態設為 down

            # 判定右腿是否抬起
            if r_ang < 100:  # 右腿抬起（角度變小代表往胸口靠近）
                if state['stage']['r'] == 'down':  # 如果右腿之前是 down
                    state['counter'] += 1  # 計數增加 1
                    state['total_calories'] += CALORIE_METRICS.get('mountain_climber', 0.15) * weight_mul  # 增加卡路里消耗
                    state['hint'] = '右腳漂亮！'  # 顯示即時鼓勵提示
                state['stage']['r'] = 'up'  # 將右腿狀態設為 up
            elif r_ang > 120:  # 右腿收回到後方（角度變大）
                state['stage']['r'] = 'down'  # 將右腿狀態設為 down

            # 提供連續的動作提示
            if l_ang > 120 and r_ang > 120:  # 兩腿都收回
                state['hint'] = '保持伏地挺身預備姿勢，左右膝輪流往胸口收'  # 提示繼續交替抬膝
            elif 100 <= l_ang <= 120 and 100 <= r_ang <= 120:  # 兩腿角度處於中間區間
                state['hint'] = '膝蓋再往胸口靠近一點，角度不足還不會計數'  # 提示增加抬膝幅度

    # ==========================================
    # 6. 深蹲 (Squat) 判定邏輯
    # ==========================================
    elif ex_type == 'squat':  # 如果運動類型是深蹲
        # 計算膝蓋的彎曲角度（髖-膝-踝）
        angle = calculate_angle(l_hip, l_kn, l_an)
        
        if angle > 160:  # 如果膝蓋打直（> 160 度，接近直線）
            if state['stage'] == 'down':  # 如果之前是蹲下狀態
                state['counter'] += 1  # 計數加 1
                state['total_calories'] += CALORIE_METRICS['squat'] * weight_mul  # 增加卡路里消耗
                state['hint'] = '站直完成一次！下一次臀部往後坐、膝蓋穩定'  # 提示下一步
            else:  # 如果不是蹲下狀態
                state['hint'] = '準備深蹲：雙腳站穩，臀部往後往下坐'  # 提示蹲下
            state['stage'] = 'up'  # 標記為站起狀態
            
        elif angle < 90:  # 如果膝蓋彎曲到接近 90 度（< 90 度）
            state['stage'] = 'down'  # 標記為蹲下狀態
            state['hint'] = '深蹲深度到位！保持胸口打開，現在站起來'  # 提示站起
            
        else:  # 其他中間角度
            state['hint'] = '再蹲低一點，膝蓋角度小於 90 度後再站起才會計數'  # 提示繼續蹲低

    # ==========================================
    # 7. 椅子三頭肌撐體 (Triceps Dip) 判定邏輯 - 垂直高度差防呆版
    # ==========================================
    elif ex_type == 'triceps_dip':  # 如果運動類型是三頭肌撐體
        # 1. 取得軀幹長度作為比例尺，避免因離鏡頭遠近造成誤差
        torso = max(abs(l_sh[1] - l_hip[1]), abs(r_sh[1] - r_hip[1]))  # 軀幹垂直長度
        if torso < 0.1:  # 如果軀幹太短（可能是數據異常）
            torso = 0.2  # 設定最小值，避免除以過小的數

        # 2. 計算肩膀與手腕的高度差比例 (Y 軸往下為正)
        # 撐起時：手腕在肩膀下方，Y 軸差距大，數值接近 1.0
        # 降下時：肩膀降至接近手腕，Y 軸差距小，數值接近 0.5
        l_dist = (l_wr[1] - l_sh[1]) / torso  # 左臂手腕相對肩膀高度
        r_dist = (r_wr[1] - r_sh[1]) / torso  # 右臂手腕相對肩膀高度
        avg_dist = (l_dist + r_dist) / 2  # 雙臂平均高度差

        # 3. 狀態機判定：用垂直距離取代容易失準的 2D 手肘角度
        if avg_dist < 0.5:  # 如果身體降下（高度差小）
            if state['stage'] != 'down':  # 如果不是已經在 down 狀態
                state['stage'] = 'down'  # 標記為降下狀態
                state['hint'] = '推起來！'  # 提示推起
                
        elif avg_dist > 0.8:  # 如果手臂撐直（高度差大）
            if state['stage'] == 'down':  # 如果之前是降下狀態
                state['counter'] += 1  # 計數加 1
                state['total_calories'] += CALORIE_METRICS['triceps_dip'] * weight_mul  # 增加卡路里消耗
                state['hint'] = '漂亮！手臂撐直完成一次，接著慢慢下壓'  # 提示下一步
            else:  # 如果不是降下狀態
                state['hint'] = '手掌撐穩椅緣，身體靠近椅子，準備往下沉'  # 提示下壓
            state['stage'] = 'up'  # 標記為撐起狀態
            
        else:  # 其他中間狀態
            state['hint'] = '繼續控制身體：下降到肩膀接近手腕，再用三頭肌推起'  # 提示繼續動作

    # ==========================================
    # 8. 原地抬膝 (High Knees) 判定邏輯
    # ==========================================
    elif ex_type == 'high_knees':  # 如果運動類型是原地抬膝
        # 判定膝蓋是否抬起至腰部高度（膝蓋 Y 座標小於髖部 Y 座標）
        l_up = l_kn[1] < l_hip[1] + 0.05  # 左膝是否抬起
        r_up = r_kn[1] < r_hip[1] + 0.05  # 右膝是否抬起

        if l_up or r_up:  # 如果至少有一隻膝蓋抬起
            if state['stage'] == 'down':  # 如果之前是放下狀態
                state['counter'] += 1  # 計數加 1
                state['total_calories'] += CALORIE_METRICS['high_knees'] * weight_mul  # 增加卡路里消耗
                state['hint'] = '做得好！膝蓋盡量抬高至腰部高度'  # 提示繼續抬高
            state['stage'] = 'up'  # 標記為抬起狀態
            
        elif not l_up and not r_up:  # 如果雙膝都放下
            state['stage'] = 'down'  # 標記為放下狀態
            # 靜止或腳放下時的純姿勢建議
            state['hint'] = '背部打直，核心收緊，左右腳快速交替高抬'  # 提示正確姿勢

    # ==========================================
    # 9. 弓步深蹲 (Lunge) 判定邏輯
    # ==========================================
    elif ex_type == 'lunge':  # 如果運動類型是弓步深蹲
        # 計算左右兩邊膝蓋的彎曲角度（髖-膝-踝）
        l_ang = calculate_angle(l_hip, l_kn, l_an)  # 左側膝蓋角度
        r_ang = calculate_angle(r_hip, r_kn, r_an)  # 右側膝蓋角度
        
        if l_ang < 110 and r_ang < 110:  # 如果雙膝都彎曲到 90 度以下（< 110 度）
            if state['stage'] == 'up':  # 如果之前是站起狀態
                state['counter'] += 1  # 計數加 1
                state['total_calories'] += CALORIE_METRICS['lunge'] * weight_mul  # 增加卡路里消耗
                state['hint'] = '弓步深度到位，完成一次！推回站姿後換腳'  # 提示推回站姿
            else:  # 如果不是站起狀態
                state['hint'] = '下蹲到位！前後腳膝蓋都彎曲，保持身體穩定'  # 提示保持穩定
            state['stage'] = 'down'  # 標記為下蹲狀態
            
        elif l_ang > 150 or r_ang > 150:  # 如果至少有一邊膝蓋打直（> 150 度）
            state['stage'] = 'up'  # 標記為站起狀態
            state['hint'] = '站姿準備完成：向前或向後跨一步，身體垂直下沉'  # 提示跨步下沉
            
        else:  # 其他中間狀態
            state['hint'] = '弓步再下沉一點，前後膝都接近彎曲後才會計數'  # 提示繼續下沉

    # ==========================================
    # 10. 掌上壓後轉身 (Pushup Rotation) 判定邏輯 - 順序防呆版
    # ==========================================
    elif ex_type == 'pushup_rotation':  # 如果運動類型是掌上壓後轉身
        # 1. 計算身體的打直程度（肩-髖-膝），用於判定是否為標準姿勢
        body_angle = calculate_angle(l_sh, l_hip, l_kn)  # 改看膝蓋，避免腳踝在畫面邊緣被裁切

        # 如果身體彎曲太嚴重（例如站著彎腰），提示姿勢不佳
        if body_angle < 120:  # 如果身體角度 < 120 度（過度彎曲）
            state['hint'] = '請保持背部與腿部呈一直線'  # 提示保持身體直線
        else:  # 身體姿勢標準
            # 2. 取得雙手肘部角度（肩-肘-腕）
            l_angle = calculate_angle(l_sh, l_el, l_wr)  # 左臂肘部角度
            r_angle = calculate_angle(r_sh, r_el, r_wr)  # 右臂肘部角度
            avg_arm_angle = (l_angle + r_angle) / 2  # 雙臂平均肘部角度

            # 3. 判斷轉身舉手動作（有一隻手腕的 Y 座標舉得比肩膀高出許多）
            is_rot = (l_wr[1] < l_sh[1] - 0.15) or (r_wr[1] < r_sh[1] - 0.15)  # 判定是否轉身舉手

            # 4. 嚴格順序狀態機：必須先「下壓」才能「舉手」
            # 階段 A：確實完成伏地挺身下壓動作（雙手平均彎曲小於 90 度）
            if avg_arm_angle <= 90:  # 如果雙臂平均角度 <= 90（手肘彎曲狀態）
                state['stage'] = 'down'  # 標記為下壓狀態
                state['hint'] = '推起並轉身舉手！'  # 提示推起並轉身

            # 階段 B：從下壓狀態撐起，並且舉手轉身
            elif state['stage'] == 'down':  # 如果之前是下壓狀態
                # 當偵測到轉身舉手時，必須保證「另一隻撐在地上的手是伸直的(>140度)」
                # 這可以完美防止「躺在地上直接舉手」或「站著亂揮手」的作弊
                if is_rot and (l_angle > 140 or r_angle > 140):  # 如果轉身舉手且另一隻手伸直
                    state['counter'] += 1  # 計數加 1
                    state['total_calories'] += CALORIE_METRICS['pushup_rotation'] * weight_mul  # 增加卡路里消耗
                    state['hint'] = '完美轉身！換邊繼續'  # 提示完成一次
                    state['stage'] = 'up'  # 結算後標記為撐起狀態，等待下一次的下壓


    return public_state(state)  # 函式結束，回傳過濾後的狀態


# ==========================================
# Flask Web API 路由定義
# ==========================================

@app.route('/')  # 定義根路由
def index():  # 主頁面路由處理函式
    return render_template('index.html')  # 回傳前端 HTML 檔案


@app.route('/healthz')  # 定義健康檢查端點（Render 平台會定期呼叫確認應用活著）
def healthz():  # 健康檢查處理函式
    return jsonify({'status': 'ok'})  # 回傳 JSON 格式的狀態


@app.route('/start_exercise', methods=['POST'])  # 定義運動開始端點（POST 方法）
def start_exercise_api():  # 運動開始 API 處理函式
    data = request.get_json(silent=True) or {}  # 解析前端傳來的 JSON 資料（失敗時返回空字典）
    ex_type = data.get('type', 'jumping_jacks')  # 取得運動類型（預設開合跳）
    level = data.get('level', 'beginner')  # 取得訓練強度（預設初階）

    # 驗證運動強度是否有效
    if level not in EXERCISE_TARGETS:  # 如果強度不在預設值內
        return jsonify({'status': 'error', 'message': 'invalid level'}), 400  # 回傳 400 錯誤
    # 驗證運動類型是否有效
    if ex_type not in EXERCISE_TARGETS[level]:  # 如果運動類型不支援該強度
        return jsonify({'status': 'error', 'message': 'invalid exercise type'}), 400  # 回傳 400 錯誤

    # 嘗試解析體重（浮點數），若失敗則使用預設值
    try:
        weight = float(data.get('weight', 65.0))  # 嘗試轉換體重為浮點數
    except (TypeError, ValueError):  # 若轉換失敗
        weight = 65.0  # 使用預設體重 65 公斤

    with state_lock:  # 獲取執行緒鎖（保護狀態修改）
        state = get_user_state()  # 取得 client 的訓練狀態

        # 更新運動狀態
        state['type'] = ex_type  # 設定運動類型
        state['level'] = level  # 設定訓練強度
        state['weight'] = weight  # 設定體重
        state['target'] = EXERCISE_TARGETS[level][ex_type]  # 設定該運動的目標次數
        state['counter'] = 0.0  # 重置計數器為 0
        state['stage'] = None  # 重置動作階段
        state['hint'] = '偵測已開始，請進入鏡頭範圍。'  # 設定提示文字
        state['last_tick'] = time.time()  # 記錄當前時間
        state['last_active'] = time.time()  # 記錄活動時間
        state['is_started'] = True  # 標記訓練已開始

        # 只在巡迴第一項開始時重置累計熱量
        if state['start_time'] is None:  # 如果是第一次開始巡迴訓練
            state['start_time'] = time.time()  # 記錄巡迴開始時間
            state['total_calories'] = 0.0  # 重置累計卡路里
            state['end_time'] = None  # 清空結束時間

        response_state = public_state(state)  # 取得可回傳給前端的狀態

    return jsonify({'status': 'success', **response_state})  # 回傳成功狀態和當前訓練狀態


@app.route('/analyze_frame', methods=['POST'])  # 定義影像分析端點（POST 方法）
def analyze_frame_api():  # 影像分析 API 處理函式
    data = request.get_json(silent=True) or {}  # 解析前端傳來的 JSON 資料

    # 快速檢查訓練是否已開始（使用臨時鎖，避免長時間持鎖）
    with state_lock:  # 獲取執行緒鎖
        state = get_user_state()  # 取得訓練狀態
        if not state.get('is_started'):  # 如果訓練還未開始
            return jsonify(public_state(state))  # 直接回傳當前狀態，不進行影像分析

    try:
        frame = decode_base64_frame(data.get('frame'))  # 解碼前端傳來的 Base64 影像
        frame = cv2.flip(frame, 1)  # 水平翻轉影像（前置鏡頭校正）
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換色彩空間 BGR -> RGB（MediaPipe 要求）

        pose_detector, pose_module, draw_utils = get_pose_detector()  # 取得姿勢偵測器
        results = pose_detector.process(img_rgb)  # 執行姿勢偵測

        # 偵測完成後再獲取鎖，進行狀態更新
        with state_lock:  # 獲取執行緒鎖
            state = get_user_state()  # 取得訓練狀態

            if not state.get('is_started'):  # 再次確認訓練是否開始（防止中斷）
                return jsonify(public_state(state))  # 回傳當前狀態

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                analyze_landmarks(landmarks, state)
                return jsonify(public_state(state, landmarks))
            else:
                state['hint'] = '未偵測到完整人體，請退後一點並保持全身入鏡。'
                return jsonify(public_state(state))

    except Exception as e:  # 若發生任何錯誤
        with state_lock:  # 獲取執行緒鎖
            state = get_user_state()  # 取得訓練狀態
            state['hint'] = f'影像分析失敗：{str(e)}'  # 設定錯誤提示
            return jsonify(public_state(state)), 400  # 回傳 400 錯誤


@app.route('/get_stats')  # 定義獲取統計數據端點
def get_stats():  # 獲取統計數據 API 處理函式
    with state_lock:  # 獲取執行緒鎖
        state = get_user_state()  # 取得訓練狀態
        return jsonify(public_state(state))  # 回傳訓練狀態


@app.route('/stop_exercise', methods=['POST'])  # 定義停止運動端點
def stop_exercise_api():  # 停止運動 API 處理函式
    with state_lock:  # 獲取執行緒鎖
        state = get_user_state()  # 取得訓練狀態
        state['is_started'] = False  # 標記訓練已停止
        state['end_time'] = time.time()  # 記錄訓練結束時間
        return jsonify({'status': 'success', **public_state(state)})  # 回傳成功狀態


@app.route('/pause_exercise', methods=['POST'])  # 定義暫停運動端點
def pause_exercise_api():  # 暫停運動 API 處理函式
    with state_lock:  # 獲取執行緒鎖
        state = get_user_state()  # 取得訓練狀態
        state['is_started'] = False  # 標記訓練已暫停
        state['hint'] = '已暫停偵測，重看教學結束後會從目前進度繼續。'  # 設定暫停提示
        return jsonify({'status': 'success', **public_state(state)})  # 回傳成功狀態


@app.route('/resume_exercise', methods=['POST'])  # 定義恢復運動端點
def resume_exercise_api():  # 恢復運動 API 處理函式
    with state_lock:  # 獲取執行緒鎖
        state = get_user_state()  # 取得訓練狀態
        state['is_started'] = True  # 標記訓練已恢復
        state['last_tick'] = time.time()  # 更新時間戳記
        state['last_active'] = time.time()  # 更新活動時間
        state['hint'] = '已恢復偵測，請回到鏡頭前繼續。'  # 設定恢復提示
        return jsonify({'status': 'success', **public_state(state)})  # 回傳成功狀態


@app.route('/reset_circuit', methods=['POST'])  # 定義重置巡迴訓練端點
def reset_circuit_api():  # 重置巡迴訓練 API 處理函式
    with state_lock:  # 獲取執行緒鎖
        client_id = get_client_id()  # 取得 client ID
        exercise_states[client_id] = create_default_state()  # 重置該 client 的訓練狀態為預設值
        return jsonify({'status': 'success', **public_state(exercise_states[client_id])})  # 回傳成功狀態


@app.route('/get_ai_feedback', methods=['POST'])  # 定義獲取 AI 教練評論端點
def get_ai_feedback():  # AI 教練評論 API 處理函式
    data = request.get_json(silent=True) or {}  # 解析前端傳來的 JSON 資料

    # 取得訓練統計資料
    with state_lock:  # 獲取執行緒鎖
        state = get_user_state()  # 取得訓練狀態
        start_time = state.get('start_time')  # 取得訓練開始時間
        end_time = state.get('end_time') or time.time()  # 取得訓練結束時間（若無則用當前時間）
        total_calories = float(state.get('total_calories', 0.0))  # 取得累計卡路里
        level = state.get('level', 'beginner')  # 取得訓練強度

    # 計算訓練時長
    if not start_time:  # 如果沒有開始時間
        duration = 0  # 時長為 0
    else:
        duration = max(0, end_time - start_time)  # 計算時長（秒），確保為非負數

    # 將秒數轉換為分秒格式
    mins, secs = int(duration // 60), int(duration % 60)  # 計算分鐘數和秒數
    # 將強度代碼轉換為中文文字
    level_text = "進階" if level == "advanced" else "初階"  # 進階或初階

    # 取得訓練完成情況
    completed_exercises = int(data.get("completed_exercises", 0))  # 完成的運動項數
    total_exercises = int(data.get("total_exercises", 12))  # 總運動項數
    skipped_exercises = int(data.get("skipped_exercises", 0))  # 跳過的項數
    was_aborted = bool(data.get("was_aborted", False))  # 是否中途放棄

    # 計算完成度比例
    completion_rate = 0.0  # 預設完成度為 0
    if total_exercises > 0:  # 如果有運動項數
        completion_rate = completed_exercises / total_exercises  # 計算完成度（0-1）

    # 根據完成度決定教練態度
    if completion_rate >= 0.9 and not was_aborted:  # 如果完成度 >= 90% 且未中途放棄
        performance_label = "完成度很高"  # 性能標籤
        coach_attitude = "可以肯定表現，但不要過度浮誇。"  # 教練態度
    elif completion_rate >= 0.5:  # 如果完成度 >= 50%
        performance_label = "完成一半以上"  # 性能標籤
        coach_attitude = "可以肯定努力，但也要指出還有很多進步空間。"  # 教練態度
    elif duration < 180 or completion_rate < 0.35 or was_aborted:  # 如果時長 < 3 分鐘或完成度 < 35% 或中途放棄
        performance_label = "太快放棄或完成度偏低"  # 性能標籤
        coach_attitude = "不要說得太好聽，可以稍微吐槽，例如「今天有點敷衍」、「這樣還不算真正練到」，但語氣不要太兇，不要人身攻擊。"  # 教練態度
    else:  # 其他情況
        performance_label = "完成度普通"  # 性能標籤
        coach_attitude = "語氣中性偏嚴格，指出問題並給下一次目標。"  # 教練態度

    # 建立 AI 教練提示詞
    prompt = f"""你現在是專業但嘴巴有點直接的居家徒手 HIIT 教練。請根據學員這次訓練狀況給最後評論。

訓練資料：
總時間：{mins}分{secs}秒
總消耗：{total_calories:.2f}大卡
訓練強度：{level_text}
完成項目：{completed_exercises}/{total_exercises}
跳過項目：{skipped_exercises}
是否中途放棄：{"是" if was_aborted else "否"}
完成度判定：{performance_label}

評論規則：
1. 用繁體中文。
2. 不要條列、不要標題、不要 markdown、不要像報告格式。
3. 請寫成自然的一小段話，像教練現場講評，約 80 到 140 字。
4. 根據完成度真實評判，不要無腦鼓勵。
5. 如果太快放棄、完成很少、熱量很低，可以直接說「太混了」、「今天有點敷衍」、「這樣還不算真正練到」這類話。
6. 如果完成度高，可以肯定表現，但還是給一個下次改進方向。
7. 絕對禁止提到啞鈴、器材、重訓或任何負重器材。
8. {coach_attitude}
"""

    # 若無 Groq API 金鑰，使用預設回應
    if client is None:  # 如果客戶端未初始化（無 API 金鑰）
        if completion_rate < 0.35 or was_aborted or duration < 180:  # 如果完成度低或中途放棄或時長太短
            fallback = (
                f"今天這輪完成 {completed_exercises}/{total_exercises} 項，時間約 {mins} 分 {secs} 秒，"
                f"消耗 {total_calories:.2f} kcal。老實說有點太混了，身體才剛開始熱起來就收工，"
                "下次至少撐過一半再說累，先把節奏穩住。"
            )
        elif completion_rate >= 0.9:  # 如果完成度高
            fallback = (
                f"這次完成 {completed_exercises}/{total_exercises} 項，時間約 {mins} 分 {secs} 秒，"
                f"消耗 {total_calories:.2f} kcal。整體完成度不錯，節奏也算穩，"
                "下次可以把動作幅度做完整一點，不要只追速度。"
            )
        else:  # 其他情況（普通完成度）
            fallback = (
                f"這次完成 {completed_exercises}/{total_exercises} 項，時間約 {mins} 分 {secs} 秒，"
                f"消耗 {total_calories:.2f} kcal。表現算有做，但還不到紮實，"
                "下次少跳過幾項，把每個動作做滿會更有效。"
            )
        return jsonify({"feedback": fallback})  # 回傳預設評論

    # 使用 Groq API 生成 AI 教練評論
    try:
        chat = client.chat.completions.create(  # 呼叫 Groq API
            messages=[{"role": "user", "content": prompt}],  # 傳遞提示詞給 AI
            model="llama-3.1-8b-instant",  # 使用 Llama 3.1 8B 模型
        )
        feedback = chat.choices[0].message.content.strip()  # 提取 AI 生成的評論文字

        # 清除開始時間，為下一個巡迴做準備
        with state_lock:  # 獲取執行緒鎖
            state = get_user_state()  # 取得訓練狀態
            state['start_time'] = None  # 清空開始時間

        return jsonify({"feedback": feedback})  # 回傳 AI 評論
    except Exception as e:  # 若發生任何錯誤
        return jsonify({"feedback": f"AI 分析失敗：{str(e)}"}), 500  # 回傳 500 錯誤


# ==========================================
# 應用程式入口點
# ==========================================
if __name__ == "__main__":  # 如果腳本作為主程式運行
    port = int(os.environ.get("PORT", "5000"))  # 從環境變數讀取 PORT，預設 5000
    app.run(debug=False, host="0.0.0.0", port=port)  # 啟動 Flask 應用程式（監聽所有網卡，生產模式無調試）
