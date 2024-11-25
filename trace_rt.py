import cv2
import mediapipe as mp
import csv
import os
import PySimpleGUI as sg
import lightgbm as lgb
from datetime import datetime
import math
import numpy as np
#2点間の距離を計算する関数
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#3点間の角度を計算する関数　(x0, y0)を原点として、(x1, y1)と(x2, y2)の角度を計算する

def calc_angle(x0, y0, x1, y1, x2, y2):
    vec1 = [x1 - x0, y1 - y0]
    vec2 = [x2 - x0, y2 - y0]
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    inner = np.inner(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)
    if np.isnan(cos_theta):  # cos_thetaがNaNだった場合は0を返す
        return 0
    cos_theta_value = cos_theta.item()
    theta = math.degrees(math.acos(cos_theta_value))
    return theta
#3点間の外積を計算する関数　(x0, y0)を原点として、(x1, y1)と(x2, y2)の外積を計算する
def calc_outerproduct(x0, y0, x1, y1, x2, y2):
    vec1 = [x1 - x0, y1 - y0]
    vec2 = [x2 - x0, y2 - y0]
    outer = np.cross(vec1, vec2)
    outer = outer*1000 #数値が小さいので1000倍しておく
    return outer
"""
#訓練済みモデルを読み込む
layout = [
    [sg.Text('Select a .pkl file')],
    [sg.Input(), sg.FileBrowse(file_types=(("Pickle Files", "*.pkl"),))],
    [sg.OK(), sg.Cancel()]
]

window = sg.Window('File Selector', layout)

event, values = window.read()
window.close()

if event == 'OK':
    model_path = values[0]
    print(model_path)
else:
    print("No file selected")
"""
#pklファイルを解凍する
import pickle
model_path = "throwdetect_trained_lgb_model_2side_20241123.pkl"
model = pickle.load(open(model_path, 'rb'))


# ファイルおよび訓練済み選択用のGUIを表示する
sg.theme('LightBlue2')
layout = [
    [sg.Text('Choose a video file:'), sg.Input(), sg.FileBrowse(key='video_file')],
    [sg.Text('Choose a save folder:'), sg.Input(), sg.FolderBrowse(key='save_folder')],
    [sg.OK(), sg.Cancel()]
]

window = sg.Window('Select File and Save Folder', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    elif event == 'OK':
        file_path = values['video_file']
        save_folder = values['save_folder']
        break

window.close()

if not file_path or not save_folder:
    exit()

# 骨格を検出するための準備
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
"""#フォルダ内の指定した拡張子を持つファイルへ操作を実行する
for filename in os.listdir(folder_path):
    if filename.endswith('.mp4') or filename.endswith('.avi'):"""
# 動画ファイルを読み込む
cap = cv2.VideoCapture(file_path)

# 出力用CSVファイルを作成する
csv_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
csv_file_path = os.path.join(save_folder, csv_file_name)
mp4_file_name = csv_file_name.replace('.csv', '.mp4')
mp4_file_path = os.path.join(save_folder, mp4_file_name)
# 出力用MP4ファイルを作成する
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(mp4_file_path, fourcc, fps, (frame_width, frame_height))

with open(csv_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['frame']
    for i in range(33):
        header.append(f'x{i}')
        header.append(f'y{i}')
        header.append(f'z{i}')
    header.append("norm_rt_forearm_dist")
    header.append("norm_rt_uparm_dist")
    header.append("norm_rt_hip_dist")
    header.append("norm_rt_knee_dist")
    header.append("norm_lt_hip_dist")
    header.append("norm_lt_knee_dist")
    header.append("rt_elbow_angle")
    header.append("rt_shoulder_angle")
    header.append("lt_elbow_angle")
    header.append("lt_shoulder_angle")
    header.append("rt_hip_angle")
    header.append("rt_knee_angle")
    header.append("lt_hip_angle")
    header.append("lt_knee_angle")
    header.append("shoulder_dist")
    header.append("hip_dist")
    header.append("shoulder_hip_ratio")
    header.append("norm_rt_elbow_size")
    header.append("norm_rt_shoulder_size")
    header.append("norm_rt_trunk_size")
    header.append("norm_lt_trunk_size")
    header.append("norm_rt_hip_size")
    header.append("norm_rt_knee_size")
    header.append("norm_lt_hip_size")
    header.append("norm_lt_knee_size")
    header.append("tilt_angle")
    header.append('throw_phase')

    writer.writerow(header)
    #black_imageを作成
    black_image = np.zeros((frame_height, frame_width, 3), np.uint8)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False,  smooth_landmarks=True, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # 骨格を検出する
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_world_landmarks:
                row = [None]
                # フレーム番号を取得する
                frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                row[0] = frame
                #black_image2を作成
                black_image2 = np.zeros((frame_height, frame_width, 3), np.uint8)
                for i in range(33):
                    row.append(results.pose_world_landmarks.landmark[i].x)
                    row.append(results.pose_world_landmarks.landmark[i].y)
                    row.append(results.pose_world_landmarks.landmark[i].z)
                
                #lightGBMに渡すパラメータを検出する
                #右肩、右肘、右手首、左肩、左肘、左手首、右股関節、右ひざ、右足関節、左股関節、左ひざ、左足関節の12点の座標を取得する
                rt_shoulder_x = results.pose_world_landmarks.landmark[12].x
                rt_shoulder_y = results.pose_world_landmarks.landmark[12].y
                rt_elbow_x = results.pose_world_landmarks.landmark[14].x
                rt_elbow_y = results.pose_world_landmarks.landmark[14].y
                rt_wrist_x = results.pose_world_landmarks.landmark[16].x
                rt_wrist_y = results.pose_world_landmarks.landmark[16].y
                lt_shoulder_x = results.pose_world_landmarks.landmark[11].x
                lt_shoulder_y = results.pose_world_landmarks.landmark[11].y
                lt_elbow_x = results.pose_world_landmarks.landmark[13].x
                lt_elbow_y = results.pose_world_landmarks.landmark[13].y
                lt_wrist_x = results.pose_world_landmarks.landmark[15].x
                lt_wrist_y = results.pose_world_landmarks.landmark[15].y
                rt_hip_x = results.pose_world_landmarks.landmark[24].x
                rt_hip_y = results.pose_world_landmarks.landmark[24].y
                rt_knee_x = results.pose_world_landmarks.landmark[26].x
                rt_knee_y = results.pose_world_landmarks.landmark[26].y
                rt_ankle_x = results.pose_world_landmarks.landmark[28].x
                rt_ankle_y = results.pose_world_landmarks.landmark[28].y
                lt_hip_x = results.pose_world_landmarks.landmark[23].x
                lt_hip_y = results.pose_world_landmarks.landmark[23].y
                lt_knee_x = results.pose_world_landmarks.landmark[25].x
                lt_knee_y = results.pose_world_landmarks.landmark[25].y
                lt_ankle_x = results.pose_world_landmarks.landmark[27].x
                lt_ankle_y = results.pose_world_landmarks.landmark[27].y
                #右手関節ー右肘間の距離を計算
                rt_forearm_dist =calculate_distance(rt_wrist_x, rt_wrist_y, rt_elbow_x, rt_elbow_y)
                #右肩ー右股関節の長さ（基準軸）を計算
                rt_trunk_dist=calculate_distance(rt_shoulder_x, rt_shoulder_y, rt_hip_x, rt_hip_y)
                #右肩ー右肘の長さ(uparm_dist)を計算
                rt_uparm_dist=calculate_distance(rt_shoulder_x, rt_shoulder_y, rt_elbow_x, rt_elbow_y)
                #右股関節ー右膝の長さ(rt_hip_dist)を計算
                rt_hip_dist=calculate_distance(rt_hip_x, rt_hip_y, rt_knee_x, rt_knee_y)
                #右膝ー右足首の長さ(rt_knee_dist)を計算
                rt_knee_dist =calculate_distance(rt_knee_x, rt_knee_y, rt_ankle_x, rt_ankle_y)
                #左股関節ー左膝の長さ(lt_hip_dist)を計算
                lt_hip_dist =calculate_distance(lt_hip_x, lt_hip_y, lt_knee_x, lt_knee_y)
                #左膝ー左足首の長さ(lt_knee_dist)を計算
                lt_knee_dist=calculate_distance (lt_knee_x, lt_knee_y, lt_ankle_x, lt_ankle_y)
                #サイズを標準化するためにforarm_dist, uparm_distをtrunk_distで割る
                norm_rt_forearm_dist=rt_forearm_dist/rt_trunk_dist
                norm_rt_uparm_dist=rt_uparm_dist/rt_trunk_dist
                norm_rt_hip_dist=rt_hip_dist/rt_trunk_dist
                norm_rt_knee_dist=rt_knee_dist/rt_trunk_dist
                norm_lt_hip_dist=lt_hip_dist/rt_trunk_dist
                norm_lt_knee_dist=lt_knee_dist/rt_trunk_dist
                #肩の幅
                shoulder_dist=calculate_distance(rt_shoulder_x, rt_shoulder_y, lt_shoulder_x, lt_shoulder_y)
                #股関節の幅
                hip_dist=calculate_distance(rt_hip_x, rt_hip_y, lt_hip_x, lt_hip_y)
                #shoulder_distをhip_distで割りshouder_hip_ratioとする
                shoulder_hip_ratio=shoulder_dist/hip_dist
                #右肩ー右肘ー右手関節のなす角度を計算 1つ目の座標が原点となる座標を入力
                rt_elbow_angle = calc_angle (rt_elbow_x, rt_elbow_y, rt_shoulder_x, rt_shoulder_y, rt_wrist_x, rt_wrist_y)
                #右肘ー右肩―右股関節のなす角度を計算
                rt_shoulder_angle = calc_angle (rt_shoulder_x, rt_shoulder_y, rt_hip_x, rt_hip_y, rt_elbow_x, rt_elbow_y)
                #左肩ー左肘ー左手関節のなす角度を計算
                lt_elbow_angle= calc_angle (lt_elbow_x, lt_elbow_y, lt_shoulder_x, lt_shoulder_y, lt_wrist_x, lt_wrist_y)
                #左肘ー左肩ー左股関節のなす角度を計算
                lt_shoulder_angle = calc_angle (lt_shoulder_x, lt_shoulder_y, lt_hip_x, lt_hip_y, lt_elbow_x, lt_elbow_y)
                #右肩ー右股関節ー右膝のなす角度を計算
                rt_hip_angle = calc_angle(rt_hip_x, rt_hip_y, rt_shoulder_x, rt_shoulder_y, rt_knee_x, rt_knee_y)
                #右股関節ー右膝ー右足首のなす角度を計算
                rt_knee_angle= calc_angle(rt_knee_x, rt_knee_y, rt_hip_x, rt_hip_y, rt_ankle_x, rt_ankle_y)
                #左股関節ー左膝ー左足首のなす角度を計算
                lt_hip_angle = calc_angle(lt_knee_x, lt_knee_y, lt_hip_x, lt_hip_y, lt_ankle_x, lt_ankle_y)
                #左股関節ー左膝ー左足首のなす角度を計算
                lt_knee_angle = calc_angle(lt_knee_x, lt_knee_y, lt_hip_x, lt_hip_y, lt_ankle_x, lt_ankle_y)

                #rt_elbow sizeは右肘ー右肩　右肘ー右手関節ベクトルの外積
                rt_elbow_size = calc_outerproduct(rt_elbow_x, rt_elbow_y, rt_shoulder_x, rt_shoulder_y, rt_wrist_x, rt_wrist_y)
                #rt_shoulder sizeは右肩ー右肘　右肩ー右股関節ベクトルの外積
                rt_shoulder_size= calc_outerproduct(rt_shoulder_x, rt_shoulder_y, rt_elbow_x, rt_elbow_y, rt_hip_x, rt_hip_y)
                #rt_trunk_sizeは右肩ー左肩　右肩ー右股関節ベクトルの外積
                rt_trunk_size = calc_outerproduct(rt_shoulder_x, rt_shoulder_y, lt_shoulder_x, lt_shoulder_y, rt_hip_x, rt_hip_y)
                #lt_trunk_sizeは左肩ー右肩　左肩ー左股関節ベクトルの外積
                lt_trunk_size = calc_outerproduct(lt_shoulder_x, lt_shoulder_y, rt_shoulder_x, rt_shoulder_y, lt_hip_x, lt_hip_y)
                #rt_hip_sizeは右股関節ー右膝　右股関節ー右肩ベクトルの外積
                rt_hip_size = calc_outerproduct(rt_hip_x, rt_hip_y, rt_knee_x, rt_knee_y, rt_shoulder_x, rt_shoulder_y)
                #rt_knee_sizeは右膝ー右足首　右膝ー右股関節ベクトルの外積
                rt_knee_size = calc_outerproduct(rt_knee_x, rt_knee_y, rt_ankle_x, rt_ankle_y, rt_hip_x, rt_hip_y)
                #lt_hip_sizeは左股関節ー左膝　左股関節ー左肩ベクトルの外積
                lt_hip_size = calc_outerproduct(lt_hip_x, lt_hip_y, lt_knee_x, lt_knee_y, lt_shoulder_x, lt_shoulder_y)
                #lt_knee_sizeは左膝ー左足首　左膝ー左股関節ベクトルの外積
                lt_knee_size = calc_outerproduct(lt_knee_x, lt_knee_y, lt_ankle_x, lt_ankle_y, lt_hip_x, lt_hip_y)
                #サイズを標準化するためにrt_elbow_size, rt_shoulder_sizeをrt_trunk_distの2乗で割る
                norm_rt_elbow_size=rt_elbow_size/(rt_trunk_dist**2)
                norm_rt_shoulder_size=rt_shoulder_size/(rt_trunk_dist**2)
                norm_rt_trunk_size=rt_trunk_size/(rt_trunk_dist**2)
                norm_lt_trunk_size=lt_trunk_size/(rt_trunk_dist**2)
                norm_rt_hip_size=rt_hip_size/(rt_trunk_dist**2)
                norm_rt_knee_size=rt_knee_size/(rt_trunk_dist**2)
                norm_lt_hip_size=lt_hip_size/(rt_trunk_dist**2)
                norm_lt_knee_size=lt_knee_size/(rt_trunk_dist**2)
                #両肩と両股関節の中心をとおる直線と垂直な直線とのなす角度を計算
                #両肩の中点を計算
                shoulder_center_x = (rt_shoulder_x + lt_shoulder_x) / 2
                shoulder_center_y = (rt_shoulder_y + lt_shoulder_y) / 2
                #両股関節の中点を計算
                hip_center_x = (rt_hip_x + lt_hip_x) / 2
                hip_center_y = (rt_hip_y + lt_hip_y) / 2
                #両肩の中点と両股関節の中点を結ぶ直線と垂直線とのなす角度を計算
                tilt_angle = np.arctan((hip_center_y - shoulder_center_y) / (hip_center_x - shoulder_center_x)) * 180 / np.pi
                tilt_angle = round(tilt_angle, 3)
                #tilt_angleの絶対値を取得
                tilt_angle = abs(tilt_angle)
                
                #機械学習モデルに渡すパラメータを出力する
                test_x=[norm_rt_forearm_dist, norm_rt_uparm_dist, norm_rt_hip_dist, norm_rt_knee_dist, norm_lt_hip_dist, norm_lt_knee_dist,  rt_elbow_angle, rt_shoulder_angle, lt_elbow_angle, lt_shoulder_angle, rt_hip_angle, rt_knee_angle, lt_hip_angle, lt_knee_angle, shoulder_hip_ratio, norm_rt_elbow_size,norm_rt_shoulder_size, norm_rt_trunk_size, norm_lt_trunk_size, norm_rt_hip_size, norm_rt_knee_size, norm_lt_hip_size, norm_lt_knee_size]
                #print(test_x)
                #lightGBMにパラメータを渡す
                #test_xをnumpyのarrayに変換
                test_x = np.array([test_x])
                #test_x=test_x.reshape(1,len(test_x))
                result = model.predict(test_x)
                row.append(norm_rt_forearm_dist)
                row.append(norm_rt_uparm_dist)
                row.append(norm_rt_hip_dist)
                row.append(norm_rt_knee_dist)
                row.append(norm_lt_hip_dist)
                row.append(norm_lt_knee_dist)
                row.append(rt_elbow_angle)
                row.append(rt_shoulder_angle)
                row.append(lt_elbow_angle)
                row.append(lt_shoulder_angle)
                row.append(rt_hip_angle)
                row.append(rt_knee_angle)
                row.append(lt_hip_angle)
                row.append(lt_knee_angle)
                row.append(shoulder_dist)
                row.append(hip_dist)
                row.append(shoulder_hip_ratio)
                row.append(norm_rt_elbow_size)
                row.append(norm_rt_shoulder_size)
                row.append(norm_rt_trunk_size)
                row.append(norm_lt_trunk_size)
                row.append(norm_rt_hip_size)
                row.append(norm_rt_knee_size)
                row.append(norm_lt_hip_size)
                row.append(norm_lt_knee_size)
                row.append(tilt_angle)
                row.append(result[0])
                writer.writerow(row)
                #print(result)

            # blacl_imageに骨格の検出結果を描画する　描画は元の画像に上書きするのでworld系の座標は用いない
            if results.pose_world_landmarks:
                color=[np.round(225*(1-result[0])),0,np.round(225*result[0])]
                color=np.asarray(color, dtype = int).tolist()
                drawing_spec = mp_drawing.DrawingSpec(color=color,thickness=4, circle_radius=2) #color=[B,G,R]で指定,線の太さと〇の色も指定可能
                mp_drawing.draw_landmarks(
                    black_image2, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
                #black_image2にframe_numberを表示する
                cv2.putText(black_image2, f'frame: {int(frame)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                """#black_imageにnorm_rt_trunk_sizeを表示する
                cv2.putText(black_image2, f'norm_rt_trunk_size: {round(norm_rt_trunk_size, 1)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                """
                #black_imageにtilt_angleを表示する
                cv2.putText(black_image2, f'tilt_angle: {round(tilt_angle, 1)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #ローカル座標での鼻・両肩・右肘・右手関節・両股関節の位置を取得する
                nose_local_x = int((results.pose_landmarks.landmark[0].x)*frame_width)
                nose_local_y = int((results.pose_landmarks.landmark[0].y)*frame_height)
                rt_shoulder_local_x = int((results.pose_landmarks.landmark[12].x)*frame_width)
                rt_shoulder_local_y = int((results.pose_landmarks.landmark[12].y)*frame_height)
                lt_shoulder_local_x = int((results.pose_landmarks.landmark[11].x)*frame_width)
                lt_shoulder_local_y = int((results.pose_landmarks.landmark[11].y)*frame_height)
                rt_elbow_local_x = int((results.pose_landmarks.landmark[14].x)*frame_width)
                rt_elbow_local_y = int((results.pose_landmarks.landmark[14].y)*frame_height)
                rt_wrist_local_x = int((results.pose_landmarks.landmark[16].x)*frame_width)
                rt_wrist_local_y = int((results.pose_landmarks.landmark[16].y)*frame_height)
                rt_hip_local_x = int((results.pose_landmarks.landmark[24].x)*frame_width)
                rt_hip_local_y = int((results.pose_landmarks.landmark[24].y)*frame_height)
                lt_hip_local_x = int((results.pose_landmarks.landmark[23].x)*frame_width)
                lt_hip_local_y = int((results.pose_landmarks.landmark[23].y)*frame_height)
                #両肩と両股関節の中点を定義
                shoulder_center_local_x = (rt_shoulder_local_x + lt_shoulder_local_x) / 2
                shoulder_center_local_y = (rt_shoulder_local_y + lt_shoulder_local_y) / 2
                hip_center_local_x = (rt_hip_local_x + lt_hip_local_x) / 2
                hip_center_local_y = (rt_hip_local_y + lt_hip_local_y) / 2

                #print(rt_shoulder_local_x, rt_shoulder_local_y, rt_elbow_local_x, rt_elbow_local_y, rt_wrist_local_x, rt_wrist_local_y)

                #black_imageに鼻・右肘・右手関節の位置を描画する
                cv2.circle(black_image, (nose_local_x, nose_local_y), 5, (0, 250, 250), -1)
                cv2.circle(black_image, (rt_elbow_local_x, rt_elbow_local_y), 5, (255, 50, 100), -1)
                cv2.circle(black_image, (rt_wrist_local_x, rt_wrist_local_y), 5, (255, 50, 200), -1)
                #black_image2に両肩の中点と両股関節の中点を結ぶ直線を描画する
                cv2.line(black_image2, (int(shoulder_center_local_x), int(shoulder_center_local_y)), (int(hip_center_local_x), int(hip_center_local_y)), (0, 0, 255), 2)
    
                
                #black_imageとblack_image2を合成する
                black_image2 = cv2.addWeighted(black_image, 0.5, black_image2, 0.5, 0)
                # 画像をMP4ファイルに書き込む
                out.write(black_image2)

            # 画像を表示する
            cv2.imshow('MediaPipe Pose', black_image2)

            if cv2.waitKey(5) & 0xFF == 27:
                break

# 後処理
cap.release()
out.release()
cv2.destroyAllWindows()
