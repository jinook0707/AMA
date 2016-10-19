import Queue
from time import time, sleep
from copy import copy
from os import path
from glob import glob

import cv2
import numpy as np

from modules.misc_funcs import get_time_stamp, writeFile, chk_msg_q, calc_pt_line_dist

flag_window = True # create an opencv window or not
flag_video_rec = False # video recording

# ======================================================

class CVProc:
    def __init__(self, parent):
        self.parent = parent

        self.contour_threshold = 1
        self.fourcc = cv2.cv.CV_FOURCC('x', 'v', 'i', 'd')
        self.video_rec = None # video recorder
        self.fSize = (960, 540) # default frame size
        self.p_rect = [215, 70, 788, 476] # rect(x1,y1,x2,y2) for defining the bottom panel of the experimental box
        self.video_rec = None
        

    # --------------------------------------------------

    def start_video_rec(self, video_fp, frame_arr):
        self.fSize= (frame_arr.shape[1], frame_arr.shape[0])
        self.video_fSize = (int(self.fSize[0]/2), int(self.fSize[1]/2)) # output video frame size
        self.video_rec = cv2.VideoWriter( video_fp, self.fourcc, self.parent.vFPS, self.video_fSize, 1 )

    # --------------------------------------------------

    def stop_video_rec(self):
        self.video_rec.release()
        self.video_rec = None

    # --------------------------------------------------
    
    def proc_img(self, frame_arr):
        result_tPos = [] # 0:head tag position, 1:tail-base tag position
        failed_to_find_tag = False
        tagSz = self.parent.tagSz
        status_msg = "%i/ %i, "%(self.parent.fi, self.parent.frame_cnt)
        #d_=None
        for i in range(2): # head and tail
            if i == 0:
                tp = self.parent.oData[self.parent.fi]["hPos"] # coordinate for the tag of the current frame
                t_col = (0,150,255) # rectangle color for head tag
                if tp[0] != None and tp[1] != None: # coordinate is already determined 
                    if tp != ('D', 'D'): # user intentionally deleted the info
                        cv2.rectangle(frame_arr, (tp[0]-tagSz/2,tp[1]-tagSz/2), (tp[0]+tagSz/2,tp[1]+tagSz/2), t_col, -1)
                    status_msg += 'H %s '%(str(tp))
                    result_tPos.append(tp)
                    continue
                tag_key = 'hPos'
                status_msg += 'H '
            elif i == 1: 
                tp = self.parent.oData[self.parent.fi]["tbPos"]
                t_col = (255,255,255) # rectangle color for tail-base tag
                if tp[0] != None and tp[1] != None: # coordinate is already determined 
                    if tp != ('D', 'D'): # user intentionally deleted the info
                        cv2.rectangle(frame_arr, (tp[0]-tagSz/2,tp[1]-tagSz/2), (tp[0]+tagSz/2,tp[1]+tagSz/2), t_col, -1)
                    status_msg += 'T %s '%(str(tp))
                    result_tPos.append(tp)
                    continue
                tag_key = 'tbPos'
                status_msg += 'T '

            if self.parent.fi > 1: pTagPos = self.parent.oData[self.parent.fi-1][tag_key] # tag position of the previous frame 
            else: pTagPos = (None, None)
            if type(pTagPos[0]) != int and type(pTagPos[1]) != int: # position info of tag is NOT available
                rect_ = (50,0,self.fSize[0],self.fSize[1]) # there's reflected  blue color spot in upper left corner when the pink wallpaper was used
            else:
                rect_ = ( int(pTagPos[0]-tagSz*1.5), int(pTagPos[1]-tagSz*1.5), int(pTagPos[0]+tagSz*1.5), int(pTagPos[1]+tagSz*1.5) ) # x1,y1,x2,y2
            
            if i == 0:
                if self.parent.sString in self.parent.blue_ht_sessions:
                    HSVmin = (110,50,50); HSVmax = (120,255,255) # head tag color (blue)
                else:
                    HSVmin = (175,100,90); HSVmax = (180,255,255) # head tag color (others are red)
            elif i == 1:
                HSVmin = (50,75,75); HSVmax = (70,255,255) # tail tag color
            
            tmp_grey_img = self.find_color(rect_, frame_arr, HSVmin, HSVmax) ### color detection for tag
            ''' 
            if i == 0:
                d_ = cv2.cvtColor(tmp_grey_img.copy(), cv2.cv.CV_GRAY2BGR)
                d_ = cv2.add(frame_arr, d_)
                cv2.rectangle(d_, (rect_[0],rect_[1]), (rect_[2],rect_[3]), (255,0,0), 2)
            '''
            wrect, rects = self.chk_contours(tmp_grey_img, self.contour_threshold)
            if len(rects) == 0: cp = (-1, -1)
            else:
                ### get median center point of detected contour rects
                cp_x = []; cp_y = []
                for r_ in rects:
                    cp_x.append( r_[0]+r_[2]/2 )
                    cp_y.append( r_[1]+r_[3]/2 )
                cp = ( int(np.median(cp_x)), int(np.median(cp_y)) )
            if cp == (-1,-1): # if the tag is not detected,
                cp = copy(pTagPos) # copy the previous tag position
                failed_to_find_tag = True
            if type(pTagPos[0]) == int and type(pTagPos[1]) == int: # prev tag position is available
                dist = np.sqrt( (cp[0]-pTagPos[0])**2 + (cp[1]-pTagPos[1])**2 ) # distance between tag positions of this frame and prev frame
                if dist > tagSz*3: cp = (-1,-1) # if tag moved too much in one frame, ignore this result 
            if type(cp[0]) == int and type(cp[1]) == int:
                cv2.rectangle(frame_arr, (cp[0]-tagSz/2,cp[1]-tagSz/2), (cp[0]+tagSz/2,cp[1]+tagSz/2), t_col, -1) # draw tag
            status_msg += '(%s,%s) '%(str(cp[0]), str(cp[1]))
            result_tPos.append(cp)

        ### draw rectangle around the arena either white or red (red when there's a tag position info is missing.)
        if failed_to_find_tag == True: col_ = (0,0,255)
        else: col_ = (255,255,255)
        cv2.rectangle(frame_arr, (self.p_rect[0],self.p_rect[1]), (self.p_rect[2],self.p_rect[3]), col_, 2)

        ### draw line to center and calculate the distance to center from head tag
        hcp_ = result_tPos[0] # head center point
        if hcp_ == (None,None) or hcp_ == ('D','D'): h2acp_dist = None
        else:
            r_ = self.p_rect
            acp_ = (r_[0]+(r_[2]-r_[0])/2,r_[1]+(r_[3]-r_[1])/2) # arena center point
            cv2.line(frame_arr, hcp_, acp_, (0,0,0), 1)
            h2acp_dist = int(round( np.sqrt((acp_[0]-hcp_[0])**2 + (acp_[1]-hcp_[1])**2) ))
            cv2.putText(frame_arr, str(h2acp_dist), (acp_[0],acp_[1]+10), cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(0,0,0), thickness=1) # write distance

        cv2.putText(frame_arr, status_msg, (10,25), cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0,250,0), thickness=2) # write status
        self.video_rec.write( cv2.resize(frame_arr,self.video_fSize) )
        #if d_ == None: 
        return frame_arr, result_tPos, h2acp_dist
        #else:
        #    return d_, result_tPos, h2acp_dist

    # --------------------------------------------------

    def find_color(self, rect, inImage, HSV_min, HSV_max):
    # Find a color(range: 'HSV_min' ~ 'HSV_max') in an area('rect') of an image('inImage')
    # 'rect' here is (x1,y1,x2,y2)
        pts_ = [ (rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1]) ] # Upper Left, Lower Left, Lower Right, Upper Right
        mask = np.zeros( (inImage.shape[0], inImage.shape[1]) , dtype=np.uint8 )
        tmp_grey_img = np.zeros( (inImage.shape[0], inImage.shape[1]) , dtype=np.uint8 )
        cv2.fillConvexPoly(mask, np.asarray(pts_), 255)
        tmp_col_img = cv2.bitwise_and(inImage, inImage, mask=mask )
        HSV_img = cv2.cvtColor(tmp_col_img, cv2.COLOR_BGR2HSV)
        tmp_grey_img = cv2.inRange(HSV_img, HSV_min, HSV_max)
        ret, tmp_grey_img = cv2.threshold(tmp_grey_img, 50, 255, cv2.THRESH_BINARY)
        return tmp_grey_img

    # --------------------------------------------------
    
    def preprocessing(self, inImage, param=[5,2,2]):
        inImage = cv2.GaussianBlur(inImage, (param[0],param[0]), 0)
        inImage = cv2.dilate(inImage, None, iterations=param[1])
        inImage = cv2.erode(inImage, None, iterations=param[2])
        return inImage

    # --------------------------------------------------

    def chk_contours(self, inImage, contour_threshold):
        contours, hierarchy = cv2.findContours(inImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        wrect = [-1,-1,-1,-1] # whole rect, bounding all the contours
        rects = [] # rects, bounding each contour piece
        for ci in xrange(len(contours)):
            #M = cv2.moments(contours[ci])
            br = cv2.boundingRect(contours[ci])
            if br[2] + br[3] > contour_threshold:
                if wrect[0] == -1 and wrect[1] == -1: wrect[0] = br[0]; wrect[1] = br[1]
                if wrect[2] == -1 and wrect[3] == -1: wrect[2] = br[0]; wrect[3] = br[1]
                if br[0] < wrect[0]: wrect[0] = br[0]
                if br[1] < wrect[1]: wrect[1] = br[1]
                if (br[0]+br[2]) > wrect[2]: wrect[2] = br[0]+br[2]
                if (br[1]+br[3]) > wrect[3]: wrect[3] = br[1]+br[3]
                rects.append(br)
        wrect[2] = wrect[2]-wrect[0]
        wrect[3] = wrect[3]-wrect[1]
        return tuple(wrect), rects
    
    # --------------------------------------------------

# ======================================================

if __name__ == '__main__':
    pass

