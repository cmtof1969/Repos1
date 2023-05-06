# 辨識對象：身體與雙手辨識與追蹤（Holistic）
import os, sys
import cv2 as cv
from mediapipe.python.solutions import drawing_utils, holistic

# 從命令列引數取得輸入影像來源及輸出影像檔名
argc = len(sys.argv)
if argc == 3 :
    # 抓取視頻，這裡以mp4檔案作為視頻來源
    # cap_video = cv.VideoCapture(sys.argv[1])
    video_source = sys.argv[1]
    # 設定輸出影像檔
    video_desti = sys.argv[2]

elif argc == 2 :
    # 抓取視頻，這裡以筆電攝像頭作為視頻來源
    # cap_video = cv.VideoCapture(0)
    video_source = 0
    # 設定輸出影像檔
    video_desti = sys.argv[1]
else :
    print("Usage: ",sys.argv[0], "[引數]")
    print("引數可以為下列兩者之一")
    print("方案一、 {輸入影像檔案名.mp4} {輸出影像檔案名.avi} ")
    print("方案二、 {輸出影像檔案名.avi} (輸入影像則擷取自筆電攝像頭) ")
    os._exit(0)

# 抓取視頻，這裡以mp4檔案作為視頻來源
# cap_video = cv.VideoCapture(".\\test.mp4")
# 抓取視頻，這裡以筆電自帶Cam作為視頻來源
# cap_video = cv.VideoCapture(0)
# 抓取視頻，以命令列引數指定的視頻來源
cap_video = cv.VideoCapture(video_source)

# 檢查是否取得視頻
if (cap_video.isOpened() == False) :
    cap_video.release()
    printf(" Incorrect video capture! ")
    os._exit(0)
else :
    cap_width = int(cap_video.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_fps = int(cap_video.get(cv.CAP_PROP_FPS))
    print("Video info:", cap_fps, "fps,", cap_width, "width,", cap_height, "height")
    # 準備寫出視頻
    out_video = cv.VideoWriter(video_desti, cv.VideoWriter_fourcc('M','J','P','G'), 10, (cap_width, cap_height))

# 定義一自變數取得對象辨識庫，辨識動態圖像
mpHolistic = holistic.Holistic(static_image_mode=False)
# 定義一自變數取得描繪方法庫，並設定描繪線條寬度等參數
mpDrawSpecLandmark = drawing_utils.DrawingSpec(thickness = 1, circle_radius = 1, color = drawing_utils.GREEN_COLOR)
mpDrawSpecConnection = drawing_utils.DrawingSpec(thickness = 1, circle_radius = 1, color = drawing_utils.RED_COLOR)

# 進行對象辨識
i = 0
while True :
    
    # 定義兩自變數，接取圖框
    # flag : 一boolean數值，表示是否有取得圖框
    # frame : 一個圖框，注意該色彩通道為BGR
    flag, frame = cap_video.read()
    
    # 這塊我加的
    # 假如圖框擷取失敗，跳出迴圈
    if not flag :
        print("cap.read() failed! i=", i)
        break
        
    # 這我加的，計數迴圈次數（圖框枚數）
    i += 1
    
    # 並使用opencv將BGR圖框轉換為RGB
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # 對RGB圖框實施對象辨識
    results = mpHolistic.process(frameRGB)
    
    # if not results.multi_hand_landmarks :
    #     continue
    
    # 針對在圖框中偵測到的單一人體對象數目
    # 定義一自變數取得偵測結果
    # for handlms in results.multi_hand_landmarks :
    # 在該圖框上描繪偵測結果
    # 描繪臉部特徵點
    # drawing_utils.draw_landmarks(frame, results.face_landmarks, holistic.FACEMESH_CONTOURS, mpDrawSpecLandmark, mpDrawSpecConnection)
    # 描繪左手特徵點
    drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, holistic.HAND_CONNECTIONS, mpDrawSpecLandmark, mpDrawSpecConnection)
    # 描繪右手特徵點
    drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, holistic.HAND_CONNECTIONS, mpDrawSpecLandmark, mpDrawSpecConnection)
    # 描繪身體姿態特徵點
    drawing_utils.draw_landmarks(frame, results.pose_landmarks, holistic.POSE_CONNECTIONS, mpDrawSpecLandmark, mpDrawSpecConnection)
    
        # print( type(handlms) )
        # print( handlms )
        # 印出hand landmarks的數據
        # for j, lm in enumerate(handlms.landmark) :
        #     print(j, lm.x, lm.y)
        
    # 使用opencv描繪該圖框中的所有偵測結果
    cv.imshow("Holistic", frame)
    # 使用opencv寫出描繪後的圖框
    out_video.write(frame)
            
    # 這塊我加的
    # 只看少數圖框
    if i >= 900 :
        print("Quit by frame number count i=", i)
        break
    
    # 以下兩行來自it coding man原有的源碼
    # opencv等待(秒數)使用者輸入按下'q'鍵
    if cv.waitKey(1) & 0xFF == ord('q') :
        print("Quit by user key!")
        break
    
# 釋放輸入視頻
cap_video.release()
# 釋放寫出視頻
out_video.release()

# 釋放opencv的所有視窗
cv.destroyAllWindows()

print("opencv released!")
print("Video info:", cap_fps, "fps,", cap_width, "width,", cap_height, "height")