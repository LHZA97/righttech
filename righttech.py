import argparse
import logging
import time
from scipy import ndimage
import sys
import cv2
import math
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, url_for, Response
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
parser.add_argument('--camera', type=str, default=0)

parser.add_argument('--resize', type=str, default='0x0',
				help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
				help='if provided, resize heatmaps before they are post-processed. default=1.0')

#parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
parser.add_argument('--show-process', type=bool, default=False,
				help='for debug purpose, if enabled, speed for inference is dropped.')

parser.add_argument('--tensorrt', type=str, default="False", help='for tensorrt process.')
parser.add_argument('--save_video', type=bool, default="False", help='To write the output')

args, unknown = parser.parse_known_args()


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

app=Flask(__name__, template_folder='templates')
@app.route('/')
def index():
  return render_template('base.html')

@app.route('/bp')
def ben():
	return render_template('bp.html')

@app.route('/sq')
def squ():
	return render_template('sq.html')

@app.route('/dl')
def dea():
	return render_template('dl.html')

@app.route('/sp')
def sho():
	return render_template('sp.html')

@app.route('/bc')
def bi():
	return render_template('bc.html')

@app.route('/te')
def tri():
	return render_template('te.html')

@app.route('/video_shoulderpress')
def video_shoulderpress():
	return Response(sp(), mimetype='multipart/x-mixed-replace; boundary=frame')

#=============== Deadlift =======================================================

@app.route('/video_deadlift_left')
def video_deadlift_left():
	return Response(dl_left(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_deadlift_right')
def video_deadlift_right():
	return Response(dl_right(), mimetype='multipart/x-mixed-replace; boundary=frame')

#=============== Bench Press ======================================================

@app.route('/video_benchpress_left')
def video_benchpress_left():
	return Response(bp_left(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_benchpress_right')
def video_benchpress_right():
	return Response(bp_right(), mimetype='multipart/x-mixed-replace; boundary=frame')

#================ Squat ============================================================

@app.route('/video_squat_left')
def video_squat_left():
	return Response(s_left(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_squat_right')
def video_squat_right():
	return Response(s_right(), mimetype='multipart/x-mixed-replace; boundary=frame')

#================ Bicep Curl ========================================================

@app.route('/video_bicepcurl_left')
def video_bicepcurl_left():
	return Response(bc_left(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_bicepcurl_right')
def video_bicepcurl_right():
	return Response(bc_right(), mimetype='multipart/x-mixed-replace; boundary=frame')

#================= Tricep Extension =================================================

@app.route('/video_tricepextension_left')
def video_tricepextension_left():
	return Response(te_left(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_tricepextension_right')
def video_tricepextension_right():
	return Response(te_right(), mimetype='multipart/x-mixed-replace; boundary=frame')

#================= End Route =========================================================

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,10)
fontScale = 0.8
white = (255, 255, 255)
black = (0, 0, 0)
neon_fuschia = (219,62,177)
neon_pink = (251,72,196)
aqua_green = (0,192,163)
comic_book_blue = (0,174,239)
aqua = (100, 255, 218)
pixie_pink = (253, 154, 251)
barely_banana = (255, 237, 153)
amber = (255,198,0)
lineType = 1.2
thickness = 3

fps_time = 0

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
#pylint: disable=too-many-arguments


def sp():

	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
	logger.debug('cam read+')

	'''cam = cv2.VideoCapture('static/sample/sp4.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	def rescale_frame(frame, percent=75):
		width = int(frame.shape[1] * percent/ 100)
		height = int(frame.shape[0] * percent/ 100)
		dim = (width, height)
		return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
	#========================== Declarartion of KeyPoints==============================

	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between Two KeyPoints ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between Three KeyPoints ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for bicep curl ===============================

	def shoulder_press_bottom( a, b, c, d, e, f):
		if a in range(45, 100) and b in range(45, 100) and c in range(120,180) and d in range(120,180) and e in range(80,120) and f in range(80,120):
			return True
		return False

	def shoulder_press_middle( a, b, c, d, e, f):
		if a in range(110,135) and b in range(110,135) and c in range(140,155) and d in range(140,155) and e in range(125,148) and f in range(125,148):
			return True
		return False

	def shoulder_press_top( a, b, c, d, e, f):
		if a in range(145,180) and b in range(145,180) and c in range(90,120) and d in range(90,120) and e in range(150,180) and f in range(150,180):
			return True
		return False


	#-------------camera--------------------------
	cam = cv2.VideoCapture(args.camera)
	#cam = cv2.VideoCapture('static/sample/sp4.mp4'))


	#====== Variables ===
	sp_bottom_count = 0
	sp_middle_count = 0
	sp_top_count = 0
	sp_bottom_increment = 0
	sp_middle_increment = 0
	sp_top_increment = 0
	sp_full_increment = 0
	prev_sp_bcount = 0
	prev_sp_mcount = 0
	prev_sp_tcount = 0
	fps_time = 0
	#====================

	global height,width
 #=======================================================================================

	
	while True:

		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		if len(humans) > 0:

			head_hand_dst_l = int(euclidian(find_point(pose, 0), find_point(pose, 7)))
			head_hand_dst_r = int(euclidian(find_point(pose, 0), find_point(pose, 4)))

			chest_elbow_left =  angle_calc(find_point(pose, 6), find_point(pose, 5), find_point(pose, 1))
			chest_elbow_right =  angle_calc(find_point(pose,3), find_point(pose,2), find_point(pose,1))
			left_arm_angle = angle_calc(find_point(pose,5), find_point(pose,6), find_point(pose,7))
			right_arm_angle = angle_calc(find_point(pose,2), find_point(pose,3), find_point(pose,4))


			bottom = shoulder_press_bottom(right_arm_angle, left_arm_angle, chest_elbow_right, chest_elbow_left, head_hand_dst_r, head_hand_dst_l)
			middle = shoulder_press_middle(right_arm_angle, left_arm_angle, chest_elbow_right, chest_elbow_left, head_hand_dst_r, head_hand_dst_l)
			top = shoulder_press_top(right_arm_angle, left_arm_angle, chest_elbow_right, chest_elbow_left, head_hand_dst_r, head_hand_dst_l)

 #=================================== Front Side Shoulder Press ===================================================    

			sp_bottom_count = 1 if bottom == True else 0
			if prev_sp_bcount - sp_bottom_count == 1:
				sp_bottom_increment +=1
			prev_sp_bcount = sp_bottom_count

			sp_middle_count = 1 if middle == True else 0
			if prev_sp_mcount - sp_middle_count == 1:
				sp_middle_increment +=1
			prev_sp_mcount = sp_middle_count

			sp_top_count = 1 if top == True else 0
			if prev_sp_tcount - sp_top_count == 1:
				sp_top_increment +=1
			prev_sp_tcount = sp_top_count

			cv2.putText(image, 'Repetition: '+ str(sp_full_increment),(10, 80),font, fontScale,amber,thickness)
			
			if sp_full_increment >= 1:
				cv2.putText(image, 'SHOULDER  PRESS',(10, 50),font,fontScale,amber,thickness)

			if sp_top_increment == False and (sp_middle_increment and sp_bottom_increment) > 0:
				sp_middle_increment = 0
				sp_bottom_increment = 0
			
			while (sp_top_increment and sp_middle_increment and sp_bottom_increment) > 0:
				sp_full_increment +=1
				sp_middle_increment = 0
				sp_bottom_increment = 0
				sp_top_increment = 0
				break 

			'''if bottom == True:
				cv2.putText(image, 'Position: Bottom',(10, 200),font, fontScale,amber,thickness)
			elif middle == True:
				cv2.putText(image, 'Position: Middle',(10, 200),font, fontScale,amber,thickness)
			elif top == True:
				cv2.putText(image, 'Position: Top',(10, 200),font, fontScale,amber,thickness)        
			else:
				cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

			cv2.putText(image, 'Repetition Bottom: '+ str(sp_bottom_increment),(10, 110),font, fontScale,amber,thickness)
			cv2.putText(image, 'Repetition Middle: '+ str(sp_middle_increment),(10, 140),font, fontScale,amber,thickness)
			cv2.putText(image, 'Repetition Top: '+ str(sp_top_increment),(10, 170),font, fontScale,amber,thickness)'''
			'''cv2.putText(image,'A:'+ str(right_arm_angle) + ', B:'+ str(left_arm_angle),(10, 170),font, fontScale,amber,2)
			cv2.putText(image,'C:' + str(chest_elbow_right) + ', D:' + str(chest_elbow_left),(10, 190),font, fontScale,amber,2)
			cv2.putText(image, 'E:' + str(head_hand_dst_r) + ', F:'+ str(head_hand_dst_l), (10, 210),font, fontScale,amber,2)'''

			
		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)
		
		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')
	
	cv2.destroyAllWindows()
	
def dl_left():
	
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')

	'''cam = cv2.VideoCapture('static/sample/dl1.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for bicep curl ===============================

	def deadlift_top(a, b, c, d, e, f):
		if a in range(160, 180) and b in range(160, 180) and (c-d <= 50) and (e-f <= 130):
			return True
		return False

	def deadlift_middle(a, b, c, d, e, f):
		if a in range(85, 140) and b in range(145, 155) and (c-d <= 50) and (e-f <= 110):
			return True
		return False

	def deadlift_bottom(a, b, c, d, e, f):
		if a in range(35, 80) and b in range(80, 140) and (c-d <= 45) and (e-f <= 90):
			return True
		return False

	#--------------camera--------------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/dl1.mp4')

	#==== Variables ======
	dl_bottom_count = 0
	dl_middle_count = 0
	dl_top_count = 0
	dl_bottom_increment = 0
	dl_middle_increment = 0
	dl_top_increment = 0
	dl_full_increment = 0
	prev_dl_bcount = 0
	prev_dl_mcount = 0
	prev_dl_tcount = 0
	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Left Side Variables ===================================

	while True:

		ret_val, image = cam.read()
		'''if ret_val==False:
			break
		image = ndimage.rotate(image, 270)'''
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		chest_knee_left = angle_calc(find_point(pose, 1), find_point(pose, 11), find_point(pose, 12))
		hip_foot_left = angle_calc(find_point(pose, 11), find_point(pose, 12), find_point(pose, 13))
	   
		dist_neck_chest = int(euclidian(find_point(pose, 0), find_point(pose, 1)))
		chest_shoulder_left = int(euclidian(find_point(pose, 1), find_point(pose, 5)))
		chest_hip_left = int(euclidian(find_point(pose, 1), find_point(pose, 11)))

		dl_left_bottom = deadlift_bottom(chest_knee_left, hip_foot_left, dist_neck_chest, chest_shoulder_left, chest_hip_left, chest_shoulder_left)
		dl_left_middle = deadlift_middle(chest_knee_left, hip_foot_left, dist_neck_chest, chest_shoulder_left, chest_hip_left, chest_shoulder_left)
		dl_left_top = deadlift_top(chest_knee_left, hip_foot_left, dist_neck_chest, chest_shoulder_left, chest_hip_left, chest_shoulder_left)


	#=============================== Left Side Deadlift =============================================================            

		dl_bottom_count = 1 if dl_left_bottom == True else 0
		if prev_dl_bcount - dl_bottom_count == 1:
			dl_bottom_increment +=1
		prev_dl_bcount = dl_bottom_count

		dl_middle_count = 1 if dl_left_middle == True else 0
		if prev_dl_mcount - dl_middle_count == 1:
			dl_middle_increment +=1
		prev_dl_mcount = dl_middle_count

		dl_top_count = 1 if dl_left_top == True else 0
		if prev_dl_tcount - dl_top_count == 1:
			dl_top_increment +=1
		prev_dl_tcount = dl_top_count

		cv2.putText(image, 'Repetition: '+ str(dl_full_increment),(10, 80),font, fontScale,barely_banana,thickness)
				
		if dl_top_increment == False and (dl_middle_increment and dl_bottom_increment) > 0:
			dl_middle_increment = 0
			dl_bottom_increment = 0
		
		if dl_full_increment >= 1:
			cv2.putText(image, 'DEADLIFT',(10, 50),font, fontScale,amber,thickness)

		while (dl_top_increment and dl_middle_increment and dl_bottom_increment) > 0:
			dl_full_increment +=1
			dl_middle_increment = 0
			dl_bottom_increment = 0
			dl_top_increment = 0
			break 

		if dl_left_bottom == True:
			cv2.putText(image, 'Position: Left Bottom',(10, 200),font, fontScale,amber,thickness)
		elif dl_left_middle == True:
			cv2.putText(image, 'Position: Left Middle',(10, 200),font, fontScale,amber,thickness)
		elif dl_left_top == True:
			cv2.putText(image, 'Position: Left Top',(10, 200),font, fontScale,amber,thickness)        
		else:
			cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

		cv2.putText(image, 'Repetition Top: '+ str(dl_top_increment),(10, 110),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Middle: '+ str(dl_middle_increment),(10, 140),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Bottom: '+ str(dl_bottom_increment),(10, 170),font, fontScale,amber,thickness)
		'''cv2.putText(image, 'A:'+ str(chest_knee_left) +', B:'+ str(hip_foot_left),(10, 150),font, fontScale,amber,2)
		cv2.putText(image, 'C:'+ str(dist_neck_chest) +', D:'+ str(chest_shoulder_left),(10, 170),font, fontScale,amber,2)
		cv2.putText(image, 'E:'+ str(chest_hip_left),(10, 190),font, fontScale,amber,2)'''


		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		
		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)
		
		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def dl_right():

	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')

	'''cam = cv2.VideoCapture('static/sample/dl1.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for deadlift ===============================

	def deadlift_top(a, b, c, d, e, f):
		if a in range(160, 180) and b in range(160, 180) and (c-d <= 50) and (e-f <= 130):
			return True
		return False

	def deadlift_middle(a, b, c, d, e, f):
		if a in range(85, 140) and b in range(145, 155) and (c-d <= 50) and (e-f <= 110):
			return True
		return False

	def deadlift_bottom(a, b, c, d, e, f):
		if a in range(35, 80) and b in range(80, 140) and (c-d <= 45) and (e-f <= 90):
			return True
		return False

	#---------Camera-------------
	#cam = cv2.VideoCapture(args.camera)
	#cam = cv2.VideoCapture('static/sample/dl1.mp4')

	#=====Variables======
	dl_bottom_count = 0
	dl_middle_count = 0
	dl_top_count = 0
	dl_bottom_increment = 0
	dl_middle_increment = 0
	dl_top_increment = 0
	dl_full_increment = 0
	prev_dl_bcount = 0
	prev_dl_mcount = 0
	prev_dl_tcount = 0


	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Right Side Variables ===================================

	while True:

		ret_val, image = cam.read()
		'''if ret_val==False:
			break
		image = ndimage.rotate(image, 270)'''
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		chest_knee_right = angle_calc(find_point(pose, 1), find_point(pose, 8), find_point(pose, 9))
		hip_foot_right = angle_calc(find_point(pose, 8), find_point(pose, 9), find_point(pose, 10))
	   
		dist_neck_chest = int(euclidian(find_point(pose, 0), find_point(pose, 1)))
		chest_shoulder_right = int(euclidian(find_point(pose, 1), find_point(pose, 2)))
		chest_hip_right = int(euclidian(find_point(pose, 1), find_point(pose, 8)))

		dl_right_bottom = deadlift_bottom(chest_knee_right, hip_foot_right, dist_neck_chest, chest_shoulder_right, chest_hip_right, chest_shoulder_right)
		dl_right_middle = deadlift_middle(chest_knee_right, hip_foot_right, dist_neck_chest, chest_shoulder_right, chest_hip_right, chest_shoulder_right)
		dl_right_top = deadlift_top(chest_knee_right, hip_foot_right, dist_neck_chest, chest_shoulder_right, chest_hip_right, chest_shoulder_right)

	#======================================= Right Side Deadlift ============================================================= 
		
		dl_bottom_count = 1 if dl_right_bottom == True else 0
		if prev_dl_bcount - dl_bottom_count == 1:
			dl_bottom_increment +=1
		prev_dl_bcount = dl_bottom_count

		dl_middle_count = 1 if dl_right_middle == True else 0
		if prev_dl_mcount - dl_middle_count == 1:
			dl_middle_increment +=1
		prev_dl_mcount = dl_middle_count

		dl_top_count = 1 if dl_right_top == True else 0
		if prev_dl_tcount - dl_top_count == 1:
			dl_top_increment +=1
		prev_dl_tcount = dl_top_count

		cv2.putText(image, 'Repetition: '+ str(dl_full_increment),(10, 80),font, fontScale,barely_banana,thickness)

		if dl_full_increment >= 1:
			cv2.putText(image, 'DEADLIFT',(10, 50),font, fontScale,amber,thickness)
				
		if dl_top_increment == False and (dl_middle_increment and dl_bottom_increment) > 0:
			dl_middle_increment = 0
			dl_bottom_increment = 0
		
		while (dl_top_increment and dl_middle_increment and dl_bottom_increment) > 0:
			dl_full_increment +=1
			dl_middle_increment = 0
			dl_bottom_increment = 0
			dl_top_increment = 0
			break 

		if dl_right_bottom == True:
			cv2.putText(image, 'Position: Right Bottom',(10, 200),font, fontScale,amber,thickness)
		elif dl_right_middle == True:
			cv2.putText(image, 'Position: Right Middle',(10, 200),font, fontScale,amber,thickness)
		elif dl_right_top == True:
			cv2.putText(image, 'Position: Right Top',(10, 200),font, fontScale,amber,thickness)        
		else:
			cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

		cv2.putText(image, 'Repetition Top: '+ str(dl_top_increment),(10, 110),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Middle: '+ str(dl_middle_increment),(10, 140),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Bottom: '+ str(dl_bottom_increment),(10, 170),font, fontScale,amber,thickness)
		'''cv2.putText(image, 'A:'+ str(chest_knee_right) +', B:'+ str(hip_foot_right),(10, 150),font, fontScale,amber,2)
		cv2.putText(image, 'C:'+ str(dist_neck_chest) +', D:'+ str(chest_shoulder_right),(10, 170),font, fontScale,amber,2)
		cv2.putText(image, 'E:'+ str(chest_hip_right),(10, 190),font, fontScale,amber,2)'''

	
		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def bp_left():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	def rescale_frame(frame, percent=75):
		width = int(frame.shape[1] * percent/ 100)
		height = int(frame.shape[0] * percent/ 100)
		dim = (width, height)
		return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
	
	
	'''cam = cv2.VideoCapture('static/sample/bp4.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for bench press ===============================

	def benchpress_top(a, b, c):
		if a in range(165, 180) and (b-c <= 110):
			return True
		return False

	def benchpress_middle(a, b, c):
		if a in range(100, 150) and (b-c <= 70):
			return True
		return False

	def benchpress_bottom(a, b, c):
		if a in range(30, 75) and (b-c <= 70):
			return True
		return False

	#-------------camera-----------------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/bp4.mp4')
	
	#====================
	bp_bottom_count = 0
	bp_middle_count = 0
	bp_top_count = 0
	bp_bottom_increment = 0
	bp_middle_increment = 0
	bp_top_increment = 0
	bp_full_increment = 0
	prev_bp_bcount = 0
	prev_bp_mcount = 0
	prev_bp_tcount = 0

	fps_time = 0
	#====================

	y1 = [0,0]
	global height,width

	#================================= While Loop && Left Side Variables ===================================

	while True:
		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		i = 1
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		if len(humans) > 0:

			angle_left_arm =  angle_calc(find_point(pose,5), find_point(pose,6), find_point(pose,7))
			neck_hip_left = int(euclidian(find_point(pose, 1), find_point(pose, 11)))
			shoulder_elbow_left = int(euclidian(find_point(pose, 5), find_point(pose, 6)))
			
			bp_left_top = benchpress_top(angle_left_arm, neck_hip_left, shoulder_elbow_left)
			bp_left_middle = benchpress_middle(angle_left_arm, neck_hip_left, shoulder_elbow_left)
			bp_left_bottom = benchpress_bottom(angle_left_arm, neck_hip_left, shoulder_elbow_left)

	#================================== Left Side Bench Press ====================================================

			for human in humans:
				for i in range(len(humans)):
					try:
						a = human.body_parts[0] #Head point
						x = a.x*image.shape[1]
						y = a.y*image.shape[0]
						y1.append(y)     
					except:
						pass
					
					cv2.putText(image, 'Repetition: '+ str(bp_full_increment),(10, 80),font, fontScale,aqua,thickness)

					if bp_full_increment >= 1:
						cv2.putText(image, 'BENCH PRESS',(10, 50),font, fontScale,amber,thickness)
						
					if bp_bottom_increment == False and (bp_middle_increment and bp_top_increment) > 0:
						bp_middle_increment = 0
						bp_top_increment = 0
						
					while (bp_top_increment and bp_middle_increment and bp_bottom_increment) > 0:
						bp_full_increment +=1
						bp_middle_increment = 0
						bp_bottom_increment = 0
						bp_top_increment = 0
						break 

					if ((y - y1[-2]) < 30) and bp_left_top == True:
						bp_top_count = 1 
						if prev_bp_tcount - bp_top_count == 1:
							bp_top_increment +=1
						prev_bp_tcount = bp_top_count
						cv2.putText(image, 'Position: Left Top', (10, 200), font, fontScale, amber,thickness)
					
					if ((y - y1[-2]) < 30) and bp_left_middle == True:
						bp_middle_count = 1 
						if prev_bp_mcount - bp_middle_count == 1:
							bp_middle_increment +=1
						prev_bp_mcount = bp_middle_count
						cv2.putText(image, 'Position: Left Middle', (10, 200), font, fontScale, amber,thickness)

					if ((y - y1[-2]) < 30) and bp_left_bottom == True:
						bp_bottom_count = 1 
						if prev_bp_bcount - bp_bottom_count == 1:
							bp_bottom_increment +=1
						prev_bp_bcount = bp_bottom_count
						cv2.putText(image, 'Position: Left Bottom', (10, 200), font, fontScale, amber,thickness)

					cv2.putText(image, 'Repetition Top: '+ str(bp_top_increment), (10, 110), font, fontScale,amber,thickness)
					cv2.putText(image, 'Repetition Middle: '+ str(bp_middle_increment), (10, 140), font, fontScale,amber,thickness)	
					cv2.putText(image, 'Repetition Bottom: '+ str(bp_bottom_increment), (10, 170), font, fontScale,amber,thickness)
					'''cv2.putText(image, 'A:'+ str(angle_left_arm), (10, 150), font, fontScale, amber, 2)
					cv2.putText(image, 'B - C:'+ str(neck_hip_left - shoulder_elbow_left), (10, 170), font, fontScale, amber,2)
					cv2.putText(image, 'y - y1[-2]:' + str(y-y1[-2]), (10, 190), font, fontScale, amber,2)'''
			
		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def bp_right():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+') 
	
	'''cam = cv2.VideoCapture('static/sample/bp4.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for bicep curl ===============================

	def benchpress_top(a, b, c):
		if a in range(165, 180) and (b-c <= 110):
			return True
		return False

	def benchpress_middle(a, b, c):
		if a in range(100, 150) and (b-c <= 70):
			return True
		return False

	def benchpress_bottom(a, b, c):
		if a in range(30, 75) and (b-c <= 70):
			return True
		return False

	#-------------camera-----------------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/bp4.mp4')
	
	#====================
	bp_bottom_count = 0
	bp_middle_count = 0
	bp_top_count = 0
	bp_bottom_increment = 0
	bp_middle_increment = 0
	bp_top_increment = 0
	bp_full_increment = 0
	prev_bp_bcount = 0
	prev_bp_mcount = 0
	prev_bp_tcount = 0

	fps_time = 0 
	#====================

	y1 = [0,0]
	global height,width

	#================================= While Loop && Left Side Variables ===================================

	while True:
		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		i = 1
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		if len(humans) > 0:

			angle_right_arm =  angle_calc(find_point(pose,4), find_point(pose,3), find_point(pose,2))
			neck_hip_right = int(euclidian(find_point(pose, 1), find_point(pose, 8)))
			shoulder_elbow_right = int(euclidian(find_point(pose, 2), find_point(pose, 3)))

			bp_right_top = benchpress_top(angle_right_arm, neck_hip_right, shoulder_elbow_right)
			bp_right_middle = benchpress_middle(angle_right_arm, neck_hip_right, shoulder_elbow_right)
			bp_right_bottom = benchpress_bottom(angle_right_arm, neck_hip_right, shoulder_elbow_right)

	#===================================== Right Side Bench Press ===========================================================

			for human in humans:
				for i in range(len(humans)):
					try:
						a = human.body_parts[0] #Head point
						x = a.x*image.shape[1]
						y = a.y*image.shape[0]
						y1.append(y)     
					except:
						pass
					
					cv2.putText(image, 'Repetition: '+ str(bp_full_increment),(10, 80),font, fontScale,aqua,thickness)

					if bp_full_increment >= 1:
						cv2.putText(image, 'BENCH PRESS',(10, 50),font, fontScale,amber,thickness)
						
					if bp_bottom_increment == False and (bp_middle_increment and bp_top_increment) > 0:
						bp_middle_increment = 0
						bp_top_increment = 0
						
					while (bp_top_increment and bp_middle_increment and bp_bottom_increment) > 0:
						bp_full_increment +=1
						bp_middle_increment = 0
						bp_bottom_increment = 0
						bp_top_increment = 0
						break 

					if ((y - y1[-2]) < 30) and bp_right_top == True:
						bp_top_count = 1 
						if prev_bp_tcount - bp_top_count == 1:
							bp_top_increment +=1
						prev_bp_tcount = bp_top_count
						cv2.putText(image, 'Position: Right Top', (10, 200), font, fontScale, amber,thickness)
					
					if ((y - y1[-2]) < 30) and bp_right_middle == True:
						bp_middle_count = 1 
						if prev_bp_mcount - bp_middle_count == 1:
							bp_middle_increment +=1
						prev_bp_mcount = bp_middle_count
						cv2.putText(image, 'Position: Right Middle', (10, 200), font, fontScale, amber,thickness)

					if ((y - y1[-2]) < 30) and bp_right_bottom == True:
						bp_bottom_count = 1 
						if prev_bp_bcount - bp_bottom_count == 1:
							bp_bottom_increment +=1
						prev_bp_bcount = bp_bottom_count
						cv2.putText(image, 'Position: Right Bottom', (10, 200), font, fontScale, amber,thickness)

					cv2.putText(image, 'Repetition Top: '+ str(bp_top_increment), (10, 110), font, fontScale,amber,thickness)
					cv2.putText(image, 'Repetition Middle: '+ str(bp_middle_increment), (10, 140), font, fontScale,amber,thickness)	
					cv2.putText(image, 'Repetition Bottom: '+ str(bp_bottom_increment), (10, 170), font, fontScale,amber,thickness)
					'''cv2.putText(image, 'A:'+ str(angle_right_arm), (10, 150), font, fontScale, amber, 2)
					cv2.putText(image, 'B - C:'+ str(neck_hip_right - shoulder_elbow_right), (10, 170), font, fontScale, amber,2)
					cv2.putText(image, 'y - y1[-2]:' + str(y-y1[-2]), (10, 190), font, fontScale, amber,2)'''
   
		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def s_left():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	'''cam = cv2.VideoCapture('static/sample/sq.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for squat ===============================

	def squat_top(a, b):
		if a in range(160, 180) and b in range(160, 180):
			return True
		return False

	def squat_middle(a, b):
		if a in range(80, 140) and b in range(145, 155):
			return True
		return False

	def squat_bottom(a, b):
		if a in range(35, 60) and b in range(80, 140):
			return True
		return False

	#-----camera----------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/sq.mp4')

	#==== Variables =====
	sq_bottom_count = 0
	sq_middle_count = 0
	sq_top_count = 0
	sq_bottom_increment = 0
	sq_middle_increment = 0
	sq_top_increment = 0
	sq_full_increment = 0
	prev_sq_bcount = 0
	prev_sq_mcount = 0
	prev_sq_tcount = 0

	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Left Side Variables ===================================
	while True:

		ret_val, image = cam.read()
		'''if ret_val==False:
			break
		image = ndimage.rotate(image, 270)'''
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		chest_knee_left = angle_calc(find_point(pose, 1), find_point(pose, 11), find_point(pose, 12))
		hip_foot_left = angle_calc(find_point(pose, 11), find_point(pose, 12), find_point(pose, 13))
	   
		dist_neck_chest = int(euclidian(find_point(pose, 0), find_point(pose, 1)))
		chest_shoulder_left = int(euclidian(find_point(pose, 1), find_point(pose, 5)))
		chest_hip_left = int(euclidian(find_point(pose, 1), find_point(pose, 11)))

		sq_left_bottom = squat_bottom(chest_knee_left, hip_foot_left)
		sq_left_middle = squat_middle(chest_knee_left, hip_foot_left)
		sq_left_top = squat_top(chest_knee_left, hip_foot_left)

	#======================================= Left Side Squat =============================================================     
		

		sq_bottom_count = 1 if sq_left_bottom == True else 0
		if prev_sq_bcount - sq_bottom_count == 1:
			sq_bottom_increment +=1
		prev_sq_bcount = sq_bottom_count

		sq_middle_count = 1 if sq_left_middle == True else 0
		if prev_sq_mcount - sq_middle_count == 1:
			sq_middle_increment +=1
		prev_sq_mcount = sq_middle_count

		sq_top_count = 1 if sq_left_top == True else 0
		if prev_sq_tcount - sq_top_count == 1:
			sq_top_increment +=1
		prev_sq_tcount = sq_top_count

		cv2.putText(image, 'Repetition: '+ str(sq_full_increment),(10, 80),font, fontScale,amber,thickness)

		if sq_full_increment >= 1:
			cv2.putText(image, 'SQUAT',(10, 50),font, fontScale,amber,thickness)

		if sq_bottom_increment == False and (sq_middle_increment and sq_top_increment) > 0:
			sq_middle_increment = 0
			sq_top_increment = 0
			
		while (sq_top_increment and sq_middle_increment and sq_bottom_increment) > 0:
			sq_full_increment +=1
			sq_middle_increment = 0
			sq_bottom_increment = 0
			sq_top_increment = 0
			break 

		'''if sq_left_bottom == True:
			cv2.putText(image, 'Position: Left Bottom',(10, 200),font, fontScale,amber,thickness)
		elif sq_left_middle == True:
			cv2.putText(image, 'Position: Left Middle',(10, 200),font, fontScale,amber,thickness)
		elif sq_left_top == True:
			cv2.putText(image, 'Position: Left Top',(10, 200),font, fontScale,amber,thickness)        
		else:
			cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

		cv2.putText(image, 'Repetition Top: '+ str(sq_top_increment),(10, 110),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Middle: '+ str(sq_middle_increment),(10, 140),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Below: '+ str(sq_bottom_increment),(10, 170),font, fontScale,amber,thickness)'''
		#cv2.putText(image, 'A:'+ str(chest_knee_left) +', B:'+ str(hip_foot_left),(10, 150),font, fontScale,amber,2)

		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		
		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def s_right():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	'''cam = cv2.VideoCapture('static/sample/sq.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#======================================== Positions for Squat =========================================

	def squat_top(a, b):
		if a in range(160, 180) and b in range(160, 180):
			return True
		return False

	def squat_middle(a, b):
		if a in range(80, 140) and b in range(145, 155):
			return True
		return False

	def squat_bottom(a, b):
		if a in range(35, 60) and b in range(80, 140):
			return True
		return False

	#-----camera----------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/sq.mp4')

	#==== Variables =====
	sq_bottom_count = 0
	sq_middle_count = 0
	sq_top_count = 0
	sq_bottom_increment = 0
	sq_middle_increment = 0
	sq_top_increment = 0
	sq_full_increment = 0
	prev_sq_bcount = 0
	prev_sq_mcount = 0
	prev_sq_tcount = 0

	fps_time = 0
	#====================
	count = 0

	global height,width
	#================================= While Loop && Right Side Variables ===================================

	while True:

		ret_val, image = cam.read()
		'''if ret_val==False:
			break
		image = ndimage.rotate(image, 270)'''
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		chest_knee_right = angle_calc(find_point(pose, 1), find_point(pose, 8), find_point(pose, 9))
		hip_foot_right = angle_calc(find_point(pose, 8), find_point(pose, 9), find_point(pose, 10))
	   
		dist_neck_chest = int(euclidian(find_point(pose, 0), find_point(pose, 1)))
		chest_shoulder_right = int(euclidian(find_point(pose, 1), find_point(pose, 2)))
		chest_hip_right = int(euclidian(find_point(pose, 1), find_point(pose, 8)))

		sq_right_bottom = squat_bottom(chest_knee_right, hip_foot_right)
		sq_right_middle = squat_middle(chest_knee_right, hip_foot_right)
		sq_right_top = squat_top(chest_knee_right, hip_foot_right)


	#======================================= Right Side Squat =============================================================     

		sq_bottom_count = 1 if sq_right_bottom == True else 0
		if prev_sq_bcount - sq_bottom_count == 1:
			sq_bottom_increment +=1
		prev_sq_bcount = sq_bottom_count

		sq_middle_count = 1 if sq_right_middle == True else 0
		if prev_sq_mcount - sq_middle_count == 1:
			sq_middle_increment +=1
		prev_sq_mcount = sq_middle_count

		sq_top_count = 1 if sq_right_top == True else 0
		if prev_sq_tcount - sq_top_count == 1:
			sq_top_increment +=1
		prev_sq_tcount = sq_top_count

		cv2.putText(image, 'Repetition: '+ str(sq_full_increment),(10, 80),font, fontScale,pixie_pink,thickness)
		
		if sq_full_increment >= 1:
			cv2.putText(image, 'SQUAT',(10, 50),font, fontScale,amber,thickness)

		if sq_bottom_increment == False and (sq_middle_increment and sq_top_increment) > 0:
			sq_middle_increment = 0
			sq_top_increment = 0
			
		while (sq_top_increment and sq_middle_increment and sq_bottom_increment) > 0:
			sq_full_increment +=1
			sq_middle_increment = 0
			sq_bottom_increment = 0
			sq_top_increment = 0
			break 

		'''if sq_right_bottom == True:
			cv2.putText(image, 'Position: Left Bottom',(10, 200),font, fontScale,amber,thickness)
		elif sq_right_middle == True:
			cv2.putText(image, 'Position: Left Middle',(10, 200),font, fontScale,amber,thickness)
		elif sq_right_top == True:
			cv2.putText(image, 'Position: Left Top',(10, 200),font, fontScale,amber,thickness)        
		else:
			cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

		cv2.putText(image, 'Repetition Top: '+ str(sq_top_increment),(10, 110),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Middle: '+ str(sq_middle_increment),(10, 140),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Below: '+ str(sq_bottom_increment),(10, 170),font, fontScale,amber,thickness)'''
		#cv2.putText(image, 'A:'+ str(chest_knee_left) +', B:'+ str(hip_foot_left),(10, 150),font, fontScale,amber,2)

		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		
		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)
		
		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
					b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		
		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def bc_left():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	'''cam = cv2.VideoCapture('static/sample/bcSide.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for bicep curl ===============================

	def bicepcurl_top(a, b, c):
		if a in range (25, 55) and (b-c <= 70):
			return True
		return False

	def bicepcurl_middle(a, b, c):
		if a in range(60, 100) and (b-c <= 70):
			return True
		return False

	def bicepcurl_bottom(a, b, c):
		if a in range(125, 170) and (b-c <= 70):
			return True
		return False

	#---------camera------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/bcSide.mp4')
	
	#====================
	bc_bottom_count = 0
	bc_middle_count = 0
	bc_top_count = 0
	bc_full_count = 0
	bc_bottom_increment = 0
	bc_middle_increment = 0
	bc_top_increment = 0
	bc_full_increment = 0
	prev_bc_bcount = 0
	prev_bc_mcount = 0
	prev_bc_tcount = 0

	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Left Side Variables ===================================


	while True:

		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		if len(humans) > 0:

			angle_left_arm =  angle_calc(find_point(pose,5), find_point(pose,6), find_point(pose,7))
			neck_hip_left = int(euclidian(find_point(pose, 1), find_point(pose, 11)))
			shoulder_elbow_left = int(euclidian(find_point(pose, 5), find_point(pose, 6)))
			
			bc_left_top = bicepcurl_top(angle_left_arm, neck_hip_left, shoulder_elbow_left)
			bc_left_middle = bicepcurl_middle(angle_left_arm, neck_hip_left, shoulder_elbow_left)
			bc_left_bottom = bicepcurl_bottom(angle_left_arm, neck_hip_left, shoulder_elbow_left)

	#==================================== Left Side Bicep Curl =============================================================

			bc_bottom_count = 1 if bc_left_bottom == True else 0
			if prev_bc_bcount - bc_bottom_count == 1:
				bc_bottom_increment +=1
			prev_bc_bcount = bc_bottom_count
		
			bc_middle_count = 1 if bc_left_middle == True else 0
			if prev_bc_mcount - bc_middle_count == 1:
				bc_middle_increment +=1
			prev_bc_mcount = bc_middle_count

			bc_top_count = 1 if bc_left_top == True else 0
			if prev_bc_tcount - bc_top_count == 1:
				bc_top_increment +=1
			prev_bc_tcount = bc_top_count

			cv2.putText(image, 'Repetition: '+ str(bc_full_increment),(10, 80),font, fontScale,black,thickness)

			if bc_full_increment >= 1:
				cv2.putText(image, 'BICEP CURL',(10, 50),font, fontScale,amber,thickness)
			
			if bc_top_increment == False and (bc_middle_increment and bc_bottom_increment) > 0:
				bc_middle_increment = 0
				bc_bottom_increment = 0
			
			while (bc_top_increment and bc_middle_increment and bc_bottom_increment) > 0:
				bc_full_increment +=1
				bc_middle_increment = 0
				bc_bottom_increment = 0
				bc_top_increment = 0
				break 


			'''if bc_left_bottom == True:
				cv2.putText(image, 'Position: Left Bottom',(10, 200),font, fontScale,amber,thickness)
			elif bc_left_middle == True:
				cv2.putText(image, 'Position: Left Middle',(10, 200),font, fontScale,amber,thickness)
			elif bc_left_top == True:
				cv2.putText(image, 'Position: Left Top',(10, 200),font, fontScale,amber,thickness)        
			else:
				cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

			cv2.putText(image, 'Repetition Top: '+ str(bc_top_increment),(10, 110),font, fontScale,amber,thickness)
			cv2.putText(image, 'Repetition Middle: '+ str(bc_middle_increment),(10, 140),font, fontScale,amber,thickness)
			cv2.putText(image, 'Repetition Below: '+ str(bc_bottom_increment),(10, 170),font, fontScale,amber,thickness)
			#cv2.putText(image, 'A:'+ str(angle_left_arm) +', B - C:'+ str(neck_hip_left - shoulder_elbow_left),(10, 150),font, fontScale,amber,2)'''
				
		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)            
		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
					b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def bc_right():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	'''cam = cv2.VideoCapture('static/sample/bcSide.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''


	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for bicep curl ===============================

	def bicepcurl_top(a, b, c):
		if a in range (25, 55) and (b-c <= 70):
			return True
		return False

	def bicepcurl_middle(a, b, c):
		if a in range(60, 100) and (b-c <= 70):
			return True
		return False

	def bicepcurl_bottom(a, b, c):
		if a in range(125, 170) and (b-c <= 70):
			return True
		return False

	#---------camera------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/bcSide.mp4')

	#====================
	bc_bottom_count = 0
	bc_middle_count = 0
	bc_top_count = 0
	bc_full_count = 0
	bc_bottom_increment = 0
	bc_middle_increment = 0
	bc_top_increment = 0
	bc_full_increment = 0
	prev_bc_bcount = 0
	prev_bc_mcount = 0
	prev_bc_tcount = 0

	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Right Side Variables ==========================================

	while True:

		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		if len(humans) > 0:

			angle_right_arm =  angle_calc(find_point(pose,4), find_point(pose,3), find_point(pose,2))
			neck_hip_right = int(euclidian(find_point(pose, 1), find_point(pose, 8)))
			shoulder_elbow_right = int(euclidian(find_point(pose, 2), find_point(pose, 3)))

			bc_right_top = bicepcurl_top(angle_right_arm, neck_hip_right, shoulder_elbow_right)
			bc_right_middle = bicepcurl_middle(angle_right_arm, neck_hip_right, shoulder_elbow_right)
			bc_right_bottom = bicepcurl_bottom(angle_right_arm, neck_hip_right, shoulder_elbow_right)
	
	#=================================== Right Side Bicep Curl =============================================================

			bc_bottom_count = 1 if bc_right_bottom == True else 0
			if prev_bc_bcount - bc_bottom_count == 1:
				bc_bottom_increment +=1 
			prev_bc_bcount = bc_bottom_count

			bc_middle_count = 1 if bc_right_middle == True else 0
			if prev_bc_mcount - bc_middle_count == 1:
				bc_middle_increment +=1
			prev_bc_mcount = bc_middle_count

			bc_top_count = 1 if bc_right_top == True else 0
			if prev_bc_tcount - bc_top_count == 1:
				bc_top_increment +=1
			prev_bc_tcount = bc_top_count

			cv2.putText(image, 'Repetition: '+ str(bc_full_increment),(10, 80),font, fontScale,black,thickness)

			if bc_full_increment >= 1:
				cv2.putText(image, 'BICEP CURL',(10, 50),font, fontScale,amber,thickness)

			if bc_top_increment == False and (bc_middle_increment and bc_bottom_increment) > 0:
				bc_middle_increment = 0
				bc_bottom_increment = 0
			
			while (bc_top_increment and bc_middle_increment and bc_bottom_increment) > 0:
				bc_full_increment +=1
				bc_middle_increment = 0
				bc_bottom_increment = 0
				bc_top_increment = 0
				break 

			'''if bc_right_bottom == True:
				cv2.putText(image, 'Position: Right Bottom',(10, 200),font, fontScale,amber,thickness)
			elif bc_right_middle == True:
				cv2.putText(image, 'Position: Right Middle',(10, 200),font, fontScale,amber,thickness)
			elif bc_right_top == True:
				cv2.putText(image, 'Position: Right Top',(10, 200),font, fontScale,amber,thickness)        
			else:
				cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,2)

			cv2.putText(image, 'Repetition Top: '+ str(bc_top_increment),(10, 110),font, fontScale,amber,thickness)
			cv2.putText(image, 'Repetition Middle: '+ str(bc_middle_increment),(10, 140),font, fontScale,amber,thickness)
			cv2.putText(image, 'Repetition Bottom: '+ str(bc_bottom_increment),(10, 170),font, fontScale,amber,thickness)
			#cv2.putText(image, 'A:'+ str(angle_right_arm) +', B - C:'+ str(neck_hip_right - shoulder_elbow_right),(10, 150),font, fontScale,amber,2)'''

		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def te_left():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	'''cam = cv2.VideoCapture('static/sample/te.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for tricep extension ===============================

	def tricep_top(a, b, c):
		if a in range(135, 175) and (b-c <= 70):
			return True
		return False

	def tricep_middle(a, b, c):
		if a in range(90,110) and (b-c <= 70):
			return True
		return False

	def tricep_bottom(a, b, c):
		if a in range(35, 60)  and (b-c <= 70):
			return True
		return False

	#---------camera------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/te.mp4')

	#====================
	te_bottom_count = 0
	te_middle_count = 0
	te_top_count = 0
	te_bottom_increment = 0
	te_middle_increment = 0
	te_top_increment = 0
	te_full_increment = 0
	prev_te_bcount = 0
	prev_te_mcount = 0
	prev_te_tcount = 0

	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Left Side Variables ===================================
	while True:

		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		angle_left_arm =  angle_calc(find_point(pose,5), find_point(pose,6), find_point(pose,7))
		neck_hip_left = int(euclidian(find_point(pose, 1), find_point(pose, 11)))
		shoulder_elbow_left = int(euclidian(find_point(pose, 5), find_point(pose, 6)))

		te_left_top = tricep_top(angle_left_arm, neck_hip_left, shoulder_elbow_left)
		te_left_middle = tricep_middle(angle_left_arm, neck_hip_left, shoulder_elbow_left)
		te_left_bottom = tricep_bottom(angle_left_arm, neck_hip_left, shoulder_elbow_left)

	#================================= Left Side Tricep Extension ========================================================

		te_bottom_count = 1 if te_left_bottom == True else 0
		if prev_te_bcount - te_bottom_count == 1:
			te_bottom_increment +=1
		prev_te_bcount = te_bottom_count

		te_middle_count = 1 if te_left_middle == True else 0
		if prev_te_mcount - te_middle_count == 1:
			te_middle_increment +=1
		prev_te_mcount = te_middle_count

		te_top_count = 1 if te_left_top == True else 0
		if prev_te_tcount - te_top_count == 1:
			te_top_increment +=1
		prev_te_tcount = te_top_count

		cv2.putText(image, 'Repetition: '+ str(te_full_increment),(10, 80),font, fontScale,amber,thickness)

		if te_full_increment >= 1:
			cv2.putText(image, 'TRICEPS EXTENSION',(10, 50),font, fontScale,amber,thickness)
			
		if te_bottom_increment == False and (te_middle_increment and te_top_increment) > 0:
			te_middle_increment = 0
			te_top_increment = 0
			
		while (te_top_increment and te_middle_increment and te_bottom_increment) > 0:
			te_full_increment +=1
			te_middle_increment = 0
			te_bottom_increment = 0
			te_top_increment = 0
			break 

		'''if te_left_bottom == True:
			cv2.putText(image, 'Position: Left Bottom',(10, 200),font, fontScale,amber,thickness)
		elif te_left_middle == True:
			cv2.putText(image, 'Position: Left Middle',(10, 200),font, fontScale,amber,thickness)
		elif te_left_top == True:
			cv2.putText(image, 'Position: Left Top',(10, 200),font, fontScale,amber,thickness)        
		else:
			cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,thickness)

		cv2.putText(image, 'Repetition Top: '+ str(te_top_increment),(10, 110),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Middle: '+ str(te_middle_increment),(10, 140),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Bottom: '+ str(te_bottom_increment),(10, 170),font, fontScale,amber,thickness)
		#cv2.putText(image, 'A:'+ str(angle_left_arm) +' B - C:'+ str(neck_hip_left - shoulder_elbow_left),(10, 150),font, fontScale,amber,2)'''
	
		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

def te_right():
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resize)
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
	else:
		e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
	logger.debug('cam read+')
	
	'''cam = cv2.VideoCapture('static/sample/te.mp4')
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))'''

	#========================== Declarartion of Points==============================
	
	def find_point(pose, p):
		for point in pose:
			try:
				body_part = point.body_parts[p]
				return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
			except:
				return(0,0)
		return (0,0)

	#========================== Distance between two points ===========================

	def euclidian(point1, point2):
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

	#========================== Angle between three points ============================

	def angle_calc(p0, p1, p2):
		try:
			a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
			b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
			c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
			angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180.0/math.pi
		except:
			return 0
		return int(angle)

	#===========================Positions for tricep extension ===============================

	def tricep_top(a, b, c):
		if a in range(135, 175) and (b-c <= 70):
			return True
		return False

	def tricep_middle(a, b, c):
		if a in range(90,110) and (b-c <= 70):
			return True
		return False

	def tricep_bottom(a, b, c):
		if a in range(35, 60)  and (b-c <= 70):
			return True
		return False

	#---------camera------------
	#cam = cv2.VideoCapture(args.camera)
	cam = cv2.VideoCapture('static/sample/te.mp4')

	#====================
	te_bottom_count = 0
	te_middle_count = 0
	te_top_count = 0
	te_bottom_increment = 0
	te_middle_increment = 0
	te_top_increment = 0
	te_full_increment = 0
	prev_te_bcount = 0
	prev_te_mcount = 0
	prev_te_tcount = 0

	fps_time = 0
	#====================

	global height,width

	#================================= While Loop && Left Side Variables ===================================

	while True:

		ret_val, image = cam.read()
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
		pose = humans
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		height,width = image.shape[0],image.shape[1]

		angle_right_arm =  angle_calc(find_point(pose,4), find_point(pose,3), find_point(pose,2))
		neck_hip_right = int(euclidian(find_point(pose, 1), find_point(pose, 8)))
		shoulder_elbow_right = int(euclidian(find_point(pose, 2), find_point(pose, 3)))

		te_right_top = tricep_top(angle_right_arm, neck_hip_right, shoulder_elbow_right)
		te_right_middle = tricep_middle(angle_right_arm, neck_hip_right, shoulder_elbow_right)
		te_right_bottom = tricep_bottom(angle_right_arm, neck_hip_right, shoulder_elbow_right)

	#================================== Right Side Tircep Extension ===================================================
			
		te_bottom_count = 1 if te_right_bottom == True else 0
		if prev_te_bcount - te_bottom_count == 1:
			te_bottom_increment +=1
		prev_te_bcount = te_bottom_count

		te_middle_count = 1 if te_right_middle == True else 0
		if prev_te_mcount - te_middle_count == 1:
			te_middle_increment +=1
		prev_te_mcount = te_middle_count

		te_top_count = 1 if te_right_top == True else 0
		if prev_te_tcount - te_top_count == 1:
			te_top_increment +=1
		prev_te_tcount = te_top_count

		cv2.putText(image, 'Repetition: '+ str(te_full_increment),(10, 80),font, fontScale,amber,thickness)

		if te_full_increment >= 1:
			cv2.putText(image, 'TRICEPS EXTENSION',(10, 50),font, fontScale,amber,thickness)
			
		if te_bottom_increment == False and (te_middle_increment and te_top_increment) > 0:
			bc_middle_increment = 0
			bc_top_increment = 0
			
		while (te_top_increment and te_middle_increment and te_bottom_increment) > 0:
			te_full_increment +=1
			te_middle_increment = 0
			te_bottom_increment = 0
			te_top_increment = 0
			break 

		'''if te_right_bottom == True:
			cv2.putText(image, 'Position: Right Bottom',(10, 200),font, fontScale,amber,2)
		elif te_right_middle == True:
			cv2.putText(image, 'Position: Right Middle',(10, 200),font, fontScale,amber,2)
		elif te_right_top == True:
			cv2.putText(image, 'Position: Right Top',(10, 200),font, fontScale,amber,2)        
		else:
			cv2.putText(image, 'Not counting',(10, 200),font, fontScale,amber,2)

		cv2.putText(image, 'Repetition Top: '+ str(te_top_increment),(10, 110),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Middle: '+ str(te_middle_increment),(10, 140),font, fontScale,amber,thickness)
		cv2.putText(image, 'Repetition Bottom: '+ str(te_bottom_increment),(10, 170),font, fontScale,amber,thickness)
		#cv2.putText(image, 'A:'+ str(angle_right_arm) +' B - C:'+ str(neck_hip_right - shoulder_elbow_right),(10, 150),font, fontScale,amber,2)'''

		logger.debug('postprocess+')
		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
		logger.debug('show+')
		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		#cv2.imshow('tf-pose-estimation result', image)

		ret, buffer = cv2.imencode('.jpg', image)
		if ret==True:
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break
		logger.debug('finished+')

	cv2.destroyAllWindows()

if __name__ == '__main__':
  	app.run(debug=True)