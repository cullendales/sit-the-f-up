import cv2
import time
import math as m
import mediapipe as mp
import os


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    
    degree = int(180/m.pi)*theta

    return degree

def notify_macos(title, message):
    os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')

def side_angle(choice):
    good_frames = 0
    bad_frames  = 0
    count = 1
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        lm = keypoints.pose_landmarks
        lmPose  = mp_pose.PoseLandmark

        # adds in error message for landmarks not being able to be detected. Usually lighting.
        if not lm:
            print("The camera cannot detect your body landmarks accurately. This may be due to lighting or camera position.")
            print("Please retry the program again.")
            return 
 
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        if offset < 100:
            cv2.putText(frame, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(frame, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        
        cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(frame, (l_ear_x, l_ear_y), 7, yellow, -1)

        cv2.circle(frame, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(frame, (l_hip_x, l_hip_y), 7, yellow, -1)
        
        cv2.circle(frame, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
        
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1
            
            cv2.putText(frame, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(frame, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(frame, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
        
        else:
            good_frames = 0
            bad_frames += 1
        
            cv2.putText(frame, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(frame, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(frame, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)   

            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames
        
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, red, 2)
        
        # added in two ways for it to respond. Will add a random chance of or mod or sth for it to have unique dialogue every so often
        # current system doesnt work as count doesnt 
        if bad_time > 5:
            if choice == "1":
                if count % 4 != 0:
                    count += 1
                    os.system('say -v Samantha "I love my wifey!"')
                    notify_macos("Posture Monitoring", "Sit the fuck up!")
                    time.sleep(7)
                else:
                    count += 1
                    os.system('say -v Samantha "Sit the fuck up, you little bitch!"')
                    notify_macos("Posture Monitoring", "Sit the fuck up!")
                    time.sleep(7)
            else:
                count += 1
                if count % 4 != 0:
                    os.system('say -v Daniel "Please sit up. Pretty Please."')
                    notify_macos("Posture Monitoring", "Please sit up. Pwetty Pwease.")
                else:
                    os.system('say -v Daniel "You look so good when you sit. Please do it for me."')
                    notify_macos("Posture Monitoring", "Please sit up. Pretty Please.")
                time.sleep(7)
        bad_time = 0

    cap.release()
    cv2.destroyAllWindows()

def front_angle(choice):
    good_frames = 0
    bad_frames  = 0
    count = 1
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        lm = keypoints.pose_landmarks
        lmPose  = mp_pose.PoseLandmark

        # adds in error message for landmarks not being able to be detected. Usually lighting.
        if not lm:
            print("The camera cannot detect your body landmarks accurately. This may be due to lighting or camera position.")
            print("Please retry the program again.")
            return 
 
        # key landmarks to track
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        mid_shldr_x = (l_shldr_x + r_shldr_x) // 2

        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].y * w)
        r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].y * h)

        nose_x = int(lm.landmark[lmPose.NOSE].x * w)

        shldr_offset = abs(l_shldr_x - r_shldr_x)
        ear_offset = abs(l_ear_x - r_ear_x)
        head_offset = abs(nose_x - mid_shldr_x)


        cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(frame, (nose_x, (l_shldr_y + r_shldr_y)//2), 7, green, -1)

        if shldr_offset < 20 and ear_offset < 20 and head_offset < 30:
            good_frames += 1
            bad_frames = 0
        else:
            bad_frames += 1
            good_frames = 0

        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames
        
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, red, 2)
        
        # added in two ways for it to respond. Will add a random chance of or mod or sth for it to have unique dialogue every so often
        # current system doesnt work as count doesnt 
        if bad_time > 2:
            if choice == "1":
                if count % 4 != 0:
                    count += 1
                    os.system('say -v Samantha "okay"')
                    notify_macos("Posture Monitoring", "Sit the fuck up!")
                    time.sleep(7)
                else:
                    count += 1
                    #os.system('say -v Samantha "Sit the fuck up, you little bitch!"')
                    os.system('say -v Samantha "okay"')
                    notify_macos("Posture Monitoring", "Sit the fuck up!")
                    time.sleep(7)
            else:
                count += 1
                if count % 4 != 0:
                    os.system('say -v Daniel "Please sit up. Pretty Please."')
                    notify_macos("Posture Monitoring", "Please sit up. Pwetty Pwease.")
                else:
                    os.system('say -v Daniel "You look so good when you sit. Please do it for me."')
                    notify_macos("Posture Monitoring", "Please sit up. Pretty Please.")
                time.sleep(7)
        bad_time = 0

    cap.release()
    cv2.destroyAllWindows()
    
    
def main():
    input_flag = False
    print("Welcome to the Posture Monitoring System!")
    print("First, are you using a camera aiming towards your side or your face?")
    while not input_flag:
        camera_angle = input("Enter 1 for side or 2 for front: ")
        if camera_angle == "1" or camera_angle == "2":
            input_flag = True
            print(f"Thank you!")
        else:
            print(f"Sorry, {camera_angle} is not an available option. Please re-enter 1 or 2.")

    print("Next, the program comes in two modes: Motivationally Mean or Sickingly Sweet")

    # can include some extra dialogue here of the mode introducing itself make it sorta rude or casual for 1
    input_flag = False
    while not input_flag:      
        choice = input("Enter 1 for Motivationally Mean or 2 for Sickingly Sweet: ")
        if choice == "1":
            input_flag = True
            print(f"You chose Motivationally Mean.")
        elif choice == "2":
            input_flag = True
            print(f"You chose Sickingly Sweet.")
        else:
            print(f"Sorry, {choice} is not an available option. Please re-enter 1 or 2.")

    if camera_angle == "1":
        side_angle(choice)
    if camera_angle == "2":
        front_angle(choice)

if __name__ == "__main__":
    main()


    