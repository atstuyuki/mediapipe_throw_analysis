## 2つの動画の特定の投球phaseを検出しimposeする
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
import cv2
import mediapipe as mp
import numpy as np
import PySimpleGUI as sg
#import TkEasyGUI as sg
import os

# MediaPipeのセットアップ
mp_pose = mp.solutions.pose
#lightGBMモデルの指定 pklファイルを解凍する
import pickle
model_path = "throwdetect_trained_lgb_model_2side_20241123.pkl"
model = pickle.load(open(model_path, 'rb'))


def process_video(video_path, color, min_detection_confidence=0.2, min_tracking_confidence=0.2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    rt_toe_x_list = []
    rt_toe_y_list = []
    trunk_tilt_angle_list = []
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            #results.pose_landmarksがある場合
            if results.pose_landmarks:
                frame_black = np.zeros(frame.shape, dtype=np.uint8)
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
                print(test_x)            
                
                #lightGBMにパラメータを渡して検出
                test_x = np.array([test_x])
                result = model.predict(test_x)   
                print(result)
                
                #result>0.8の場合にframe_blackに骨格を描画し、trunk_tilt_angle_listにtrunk_tilt_angleを追加　それ以外で描画しない
                if result>0.8:
                    mp.solutions.drawing_utils.draw_landmarks(
                    frame_black, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2))
                    #両肩の中点と両股関節の中点の座標を取得し、その座標をむすぶ直線を描画
                    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                    shoulder_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                    shoulder_center_y = int((left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0])
                    hip_center_x = int((left_hip.x + right_hip.x) / 2 * frame.shape[1])
                    hip_center_y = int((left_hip.y + right_hip.y) / 2 * frame.shape[0])
                    cv2.line(frame_black, (shoulder_center_x, shoulder_center_y), (hip_center_x, hip_center_y), color, 2)
                    #trunk_tilt_angleとして肩の中点と股関節の中点をむすぶ直線の傾きを計算して弧度法で表示
                    trunk_tilt_angle = np.arctan2(hip_center_y - shoulder_center_y, hip_center_x - shoulder_center_x)
                    trunk_tilt_angle = np.rad2deg(trunk_tilt_angle)
                    
                    #trunk_tilt_angle_listにtrunk_tilt_angleを追加
                    trunk_tilt_angle_list.append(trunk_tilt_angle)
                                
                    frames.append(frame_black)
                    rt_toe_x_list.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                    rt_toe_y_list.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)

        cap.release()
        return frames, rt_toe_x_list, rt_toe_y_list, trunk_tilt_angle_list

def combine_videos(frames1, frames2, rt_toe_x_list1, rt_toe_y_list1, rt_toe_x_list2, rt_toe_y_list2, trunk_tilt_angle_list1, trunk_tilt_angle_list2):
    h, w, _ = frames1[0].shape
    combined_frames = []
    for frame1, frame2, rt_toe_x1, rt_toe_y1, rt_toe_x2, rt_toe_y2,trunk_tilt_angle1, trunk_tilt_angle2 in zip(frames1, frames2, rt_toe_x_list1, rt_toe_y_list1, rt_toe_x_list2, rt_toe_y_list2, trunk_tilt_angle_list1, trunk_tilt_angle_list2):
        if rt_toe_x1 is not None and rt_toe_y1 is not None and rt_toe_x2 is not None and rt_toe_y2 is not None:
            offset_x = int(rt_toe_x1 * w - rt_toe_x2 * w)        
            offset_y = int(rt_toe_y1 * h - rt_toe_y2 * h)
            #cv2.rapAffineを使ってfrmae2をoffset_x, offset_yだけ平行移動
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            frame2 = cv2.warpAffine(frame2, M, (w, h))
            #frame1とframe2を合成
            combined_frame = cv2.addWeighted(frame1, 1, frame2, 1, 0)
            #combined_frameに1st_movie RED, 2nd_movie BLUEをcv2.putTextで表示
            cv2.putText(combined_frame, '1st_movie', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_frame, '2nd_movie', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #combined_frameにtrunk_tilt_angleをcv2.putTextで表示
            cv2.putText(combined_frame, 'trunk_tilt_angle: {:.2f}'.format(trunk_tilt_angle1), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_frame, 'trunk_tilt_angle: {:.2f}'.format(trunk_tilt_angle2), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            
        else:
            combined_frame = frame1
        combined_frames.append(combined_frame)
    return combined_frames

def select_videos_and_folder():
    layout = [
        [sg.Text('Select the first video file'), sg.Input(), sg.FileBrowse()],
        [sg.Text('Select the second video file'), sg.Input(), sg.FileBrowse()],
        [sg.Text('Select the folder to save the combined video'), sg.Input(), sg.FolderBrowse()],
        [sg.OK(), sg.Cancel()]
    ]
    window = sg.Window('Select Videos and Save Folder', layout)
    event, values = window.read()
    window.close()
    if event == 'OK':
        return values[0], values[1], values[2]
    else:
        return None, None, None

# 動画ファイルのパスと保存フォルダをPySimpleGUIで指定
video_path1, video_path2, save_folder = select_videos_and_folder()

if video_path1 and video_path2 and save_folder:
    # 各動画の処理（1つ目の動画は赤、2つ目の動画は青で骨格検出）
    frames1, rt_toe_x_list1, rt_toe_y_list1, trunk_tilt_angle_list1 = process_video(video_path1, (0, 0, 255), min_detection_confidence=0.7, min_tracking_confidence=0.7)
    frames2, rt_toe_x_list2, rt_toe_y_list2, trunk_tilt_angle_list2 = process_video(video_path2, (255, 0, 0), min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # frames1とframes2の長さを長い方に合わせて、短い方の最後のフレームを追加
    if len(frames1) > len(frames2):
        for i in range(len(frames1) - len(frames2)):
            frames2.append(frames2[-1])
            rt_toe_x_list2.append(rt_toe_x_list2[-1])
            rt_toe_y_list2.append(rt_toe_y_list2[-1])
            trunk_tilt_angle_list2.append(trunk_tilt_angle_list2[-1])
            
    else:
        for i in range(len(frames2) - len(frames1)):
            frames1.append(frames1[-1])
            rt_toe_x_list1.append(rt_toe_x_list1[-1])
            rt_toe_y_list1.append(rt_toe_y_list1[-1])
            trunk_tilt_angle_list1.append(trunk_tilt_angle_list1[-1])

    # 骨格を描画したフレームを組み合わせ
    combined_frames = combine_videos(frames1, frames2, rt_toe_x_list1, rt_toe_y_list1, rt_toe_x_list2, rt_toe_y_list2, trunk_tilt_angle_list1, trunk_tilt_angle_list2)

    # 結果を保存 ファイル名は1つ目と2つ目の動画のファイル名を結合
    output_path = os.path.join(save_folder, video_path1.split('/')[-1].split('.')[0] + '_' + video_path2.split('/')[-1].split('.')[0] + '.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames1[0].shape[1], frames1[0].shape[0]))
    for frame in combined_frames:
        out.write(frame)
    out.release()

    print("処理が完了しました。" + output_path + "を確認してください。")
else:
    print("動画ファイルまたは保存フォルダが選択されていません。")
