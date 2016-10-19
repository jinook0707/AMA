# coding: UTF-8

'''
Alligator Motion Analysis v.0.1
jinook.oh@univie.ac.at
Cognitive Biology Dept., University of Vienna
- 2016.01

----------------------------------------------------------------------
Copyright (C) 2015 Jinook Oh, W. Tecumseh Fitch 
- Contact: jinook.oh@univie.ac.at, tecumseh.fitch@univie.ac.at

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
----------------------------------------------------------------------

1) Left mouse click : Set the head tag location
- if click happens right on the already recognised head tag, 
the head tag will be deleted from the frame. 
This deletion is necessary especially for Sh_1 and 2.
Let’s say an alligator was walking around and then walked 
into the shelter. Then tag disappears from the scene. 
When the color detection can’t find the tag location, 
it just keeps the previous location as the current location. 
It’s useful to reduce user’s clicking, but in this shelter case, 
it’s wrong info. So the user should delete the tag when the 
subject disappears into the shelter.
            
2) Right mouse click : Set the tail tag location 
- Deletion function is as same as left mouse click.

3) Left arrow key : Move to previous frame (-1)
4) Shift + Left : Move to a frame (-60; one second)
5) Cmd + Left : Move to a frame (-1000)

6) Right arrow key : Move to next frame (+1)
7) Shift + Right : Move to a frame (+60)
8) Cmd + Right : Move to a frame (+1000)

9) J : Move the ‘rect’ left
10) L : Move the ‘rect’ right
11) I : Move the ‘rect’ up
12) K : Move the ‘rect’ down

13) Shift + J : Reduce width of ‘rect’
14) Shift + L : Increase width of ‘rect’
15) Shift + I : Reduce height of ‘rect’
16) Shift + K : Increase height of ‘rect’
'''

import Queue, plistlib
from threading import Thread
from os import getcwd, path
from sys import argv
from copy import copy
from time import time, sleep
from datetime import timedelta
from glob import glob
from random import shuffle

import wx
import numpy as np

from modules.misc_funcs import GNU_notice, get_time_stamp, writeFile, show_msg, load_img, cvImg_to_wxBMP, calc_angle_diff  
from modules.cv_proc import CVProc

# ======================================================

class AMAFrame(wx.Frame):
    def __init__(self):
        self.w_size = (970, 670)
        wx.Frame.__init__(self, None, -1, 'AMA', size=self.w_size) # init frame
        self.SetPosition( (30, 30) )
        self.Show(True)
        self.panel = wx.Panel(self, pos=(0,0), size=self.w_size)
        self.panel.SetBackgroundColour('#000000')
        self.cv_proc = CVProc(self) 

        ### init variables
        self.msg_q = Queue.Queue()
        self.program_start_time = time()
        self.session_start_time = -1
        self.oData = {} # output data
        self.fPath = '' # folder path including frame images
        self.sString = '' # session string such as '287_Sh_1', '289_NE_1', and so on..
        self.blue_ht_sessions = ['286_Sh_2', '287_NE_2', '288_Sh_2', '289_Sh_2', '290_Sh_2', '291_Sh_1', '292_Sh_1', '293_Sh_1', '294_Sh_1', '295_NE_1', '296_NE_1', '297_Sh_1', '298_Sh_2', '299_Sh_1', '300_NE_1', '301_NE_1', '302_NE_2', '303_NE_2', '304_NE_2', '305_NE_2', '306_Sh_0', '306_Sh_1', '307_Sh_1'] # head tag color is blue in these sessions (red in others)
        self.fi = 0 # current frame index
        self.frame_cnt = 0
        self.vFPS = 60 # fps for video file
        self.tagSz = 10 # head/tail_base tag size (length of one edge of square shape)
        self.is_running = False # analysis is running by pressing spacebar
        self.timer_run = None # timer for running analysis

        ### user interface setup
        posX = 5
        posY = 10
        btn_width = 150
        b_space = 30
        self.btn_start = wx.Button(self.panel, -1, label='Analyze video (folder)', pos=(posX,posY), size=(btn_width, -1))
        self.btn_start.Bind(wx.EVT_LEFT_UP, self.onStartStopAnalyzeVideo)
        posY += b_space
        self.btn_quit = wx.Button(self.panel, -1, label='QUIT', pos=(posX,posY), size=(btn_width, -1))
        self.btn_quit.Bind(wx.EVT_LEFT_UP, self.onClose)
        
        ### Elapsed time
        posX = 170
        posY = 15
        self.sTxt_pr_time = wx.StaticText(self.panel, -1, label='0:00:00', pos=(posX, posY)) # elapsed time since program starts
        _x = self.sTxt_pr_time.GetPosition()[0] + self.sTxt_pr_time.GetSize()[0] + 15
        _stxt = wx.StaticText(self.panel, -1, label='since program started', pos=(_x, posY))
        _stxt.SetForegroundColour('#CCCCCC')
        self.font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.NORMAL)
        self.sTxt_pr_time.SetFont(self.font)
        self.sTxt_pr_time.SetBackgroundColour('#000000')
        self.sTxt_pr_time.SetForegroundColour('#00FF00')
        posY += b_space
        self.sTxt_s_time = wx.StaticText(self.panel, -1, label='0:00:00', pos=(posX, posY)) # elapsed time since session starts
        _x = self.sTxt_s_time.GetPosition()[0] + self.sTxt_s_time.GetSize()[0] + 15
        _stxt = wx.StaticText(self.panel, -1, label='since session started', pos=(_x, posY))
        _stxt.SetForegroundColour('#CCCCCC')
        self.sTxt_s_time.SetFont(self.font)
        self.sTxt_s_time.SetBackgroundColour('#000000')
        self.sTxt_s_time.SetForegroundColour('#CCCCFF')
       
        posX = _stxt.GetPosition()[0] + _stxt.GetSize()[0] + 50
        posY = self.sTxt_pr_time.GetPosition()[1]
        self.sTxt_fps = wx.StaticText(self.panel, -1, label='', pos=(posX, posY)) # FPS
        self.sTxt_fps.SetForegroundColour('#CCCCCC')
        posX = _stxt.GetPosition()[0] + _stxt.GetSize()[0] + 50
        posY = self.sTxt_s_time.GetPosition()[1]
        self.sTxt_fp = wx.StaticText(self.panel, -1, label='', pos=(posX, posY)) # folder path
        self.sTxt_fp.SetForegroundColour('#CCCCCC')
        '''
        self.txt_fr = wx.TextCtrl(self.panel, id=-1, pos= (posX, posY-2), value='0', size=(60, -1), style=wx.TE_PROCESS_ENTER|wx.TE_RIGHT) 
        self.txt_fr.SetMaxLength(6)
        self.txt_fr.Bind(wx.EVT_TEXT_ENTER, self.onFrameEntered)
        posX += self.txt_fr.GetSize()[0] + 10
        _stxt = wx.StaticText(self.panel, -1, label='Input frame number and <Enter> to jump to that frame.', pos=(posX,posY))
        _stxt.SetForegroundColour('#CCCCCC')
        '''
        
        ### Frame image
        self.loaded_img_pos = (5, self.sTxt_fp.GetPosition()[1]+self.sTxt_fp.GetSize()[1]+20)
        self.loaded_img = wx.StaticBitmap( self.panel, -1, wx.NullBitmap, self.loaded_img_pos, (5,5) )
        self.loaded_img.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None) 
        self.loaded_img.Bind(wx.EVT_LEFT_UP, self.onMouseLeftUp)
        self.loaded_img.Bind(wx.EVT_RIGHT_UP, self.onMouseRightUp)
        
        statbar = wx.StatusBar(self, -1)
        self.SetStatusBar(statbar)

        ### keyboard binding
        exit_btnId = wx.NewId()
        save_btnId = wx.NewId()
        left_btnId = wx.NewId(); right_btnId = wx.NewId()
        leftJump_btnId = wx.NewId(); rightJump_btnId = wx.NewId()
        leftJumpFurther_btnId = wx.NewId(); rightJumpFurther_btnId = wx.NewId()
        moveRectUp_btnId = wx.NewId(); moveRectDown_btnId = wx.NewId(); moveRectLeft_btnId = wx.NewId(); moveRectRight_btnId = wx.NewId() # p_rect of cv_proc
        resizeRectUp_btnId = wx.NewId(); resizeRectDown_btnId = wx.NewId(); resizeRectLeft_btnId = wx.NewId(); resizeRectRight_btnId = wx.NewId()
        space_btnId = wx.NewId() 
        self.Bind(wx.EVT_MENU, self.onClose, id = exit_btnId)
        self.Bind(wx.EVT_MENU, self.onSave, id = save_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onLeft(event, 'left'), id=left_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onLeft(event, 'leftjump'), id=leftJump_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onLeft(event, 'leftjumpfurther'), id=leftJumpFurther_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onRight(event, 'right'), id=right_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onRight(event, 'rightjump'), id=rightJump_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onRight(event, 'rightjumpfurther'), id=rightJumpFurther_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 'm_left'), id=moveRectLeft_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 'm_right'), id=moveRectRight_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 'm_up'), id=moveRectUp_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 'm_down'), id=moveRectDown_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 's_left'), id=resizeRectLeft_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 's_right'), id=resizeRectRight_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 's_up'), id=resizeRectUp_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onAdjustRect(event, 's_down'), id=resizeRectDown_btnId)
        self.Bind(wx.EVT_MENU, self.onSpace, id = space_btnId)
        accel_tbl = wx.AcceleratorTable([ (wx.ACCEL_CMD,  ord('Q'), exit_btnId ), 
                                          (wx.ACCEL_CMD,  ord('S'), save_btnId ),
                                          (wx.ACCEL_NORMAL,  wx.WXK_RIGHT, right_btnId ), 
                                          (wx.ACCEL_NORMAL,  wx.WXK_LEFT, left_btnId ), 
                                          (wx.ACCEL_SHIFT,  wx.WXK_RIGHT, rightJump_btnId ), 
                                          (wx.ACCEL_SHIFT,  wx.WXK_LEFT, leftJump_btnId ),
                                          (wx.ACCEL_NORMAL, wx.WXK_SPACE, space_btnId),
                                          (wx.ACCEL_CMD, wx.WXK_LEFT, leftJumpFurther_btnId), 
                                          (wx.ACCEL_NORMAL,  ord('J'), moveRectLeft_btnId ), 
                                          (wx.ACCEL_NORMAL,  ord('L'), moveRectRight_btnId ), 
                                          (wx.ACCEL_NORMAL,  ord('I'), moveRectUp_btnId ), 
                                          (wx.ACCEL_NORMAL,  ord('K'), moveRectDown_btnId ), 
                                          (wx.ACCEL_SHIFT,  ord('J'), resizeRectLeft_btnId ), 
                                          (wx.ACCEL_SHIFT,  ord('L'), resizeRectRight_btnId ), 
                                          (wx.ACCEL_SHIFT,  ord('I'), resizeRectUp_btnId ), 
                                          (wx.ACCEL_SHIFT,  ord('K'), resizeRectDown_btnId ), 
                                          (wx.ACCEL_CMD, wx.WXK_RIGHT, rightJumpFurther_btnId) ]) 
        self.SetAcceleratorTable(accel_tbl)
        
        ### set timer for processing message and updating the current running time
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimer, self.timer)
        self.timer.Start(100)

        self.Bind( wx.EVT_CLOSE, self.onClose )

    # --------------------------------------------------       

    def onFrameEntered(self, event):
        ''' jump to a specific frame
        '''
        if self.fPath == '':
            obj.SetValue( '1' )
            return
        if self.is_running == True and event != None: # if continuous analysis is running and user pressed this manual navigation key
            self.onSpace(None) # stop it
        obj = event.GetEventObject()
        fi_ = obj.GetValue()
        if fi_ < 1 or fi_ > self.frame_cnt:
            obj.SetValue( str(self.fi) )
            return
        else:
            self.fi = fi_
            self.proc_img()

    # --------------------------------------------------       
      
    def onLeft(self, event, flag):
        ''' left arrow key is pressed. go backward in the series of images
        '''
        if self.fPath == '' or self.fi == 1: return
        if self.is_running == True: # if continuous analysis is running
            if event != None: # if user pressed this manual navigation key
                self.onSpace(None) # stop continous analysis
        if flag == 'left': self.fi -= 1
        elif flag == 'leftjump': self.fi = max(1, self.fi-self.vFPS)
        elif flag == 'leftjumpfurther': self.fi = max(1, self.fi-1000)
        self.proc_img()

    # --------------------------------------------------       
    
    def onRight(self, event, flag):
        ''' right arrow key is pressed. go forward in the series of images
        '''
        if self.fPath == '' or self.fi >= self.frame_cnt: return
        if self.is_running == True: # if continuous analysis is running 
            ### FPS update
            self.fps += 1
            if time()-self.last_fps_time >= 1:
                self.sTxt_fps.SetLabel( "FPS: %i"%(self.fps) )
                self.fps = 0
                self.last_fps_time = time()
            if event != None: # user pressed this manual navigation key
                self.onSpace(None) # stop continuous analysis
        if flag == 'right': self.fi += 1
        elif flag == 'rightjump': self.fi = min(self.frame_cnt, self.fi+self.vFPS)
        elif flag == 'rightjumpfurther': self.fi = min(self.frame_cnt, self.fi+1000) 
        if self.fi == self.frame_cnt and self.is_running == True: self.onSpace(None)
        self.proc_img()

    #------------------------------------------------
    
    def onSpace(self, event):
        ''' start/stop continuous frame analysis
        '''
        if self.fPath == '' or self.fi > self.frame_cnt: return
        if self.is_running == False:
            self.is_running = True
            self.fps = 0
            self.last_fps_time = time()
            self.timer_run = wx.FutureCall(1, self.onRight, None, 'right')
        else:
            try: # stop timer
                self.timer_run.Stop() 
                self.timer_run = None
            except: pass
            self.sTxt_fps.SetLabel('')
            self.is_running = False # stop continuous analysis
            
    #------------------------------------------------
    
    def onMouseLeftUp(self, event):
        if self.fPath == '': return
        mp = event.GetPosition()
        p_ = self.oData[self.fi]['hPos']
        if p_ == (None,None) or p_ == ('D','D'): # position info is not determined or intentionally deleted 
            self.oData[self.fi]['hPos'] = (mp[0], mp[1])
        else:
            r_ = (p_[0]-self.tagSz/2, p_[1]-self.tagSz/2, p_[0]+self.tagSz/2, p_[1]+self.tagSz/2) # x1,y1,x2,y2
            if r_[0] <= mp[0] <= r_[2] and r_[1] <= mp[1] <= r_[3]: # mouse clicked in the tag area
                self.oData[self.fi]['hPos'] = ('D','D') # delete info
            else:
                self.oData[self.fi]['hPos'] = (mp[0], mp[1])
        self.proc_img()

    #------------------------------------------------
 
    def onMouseRightUp(self, event):
        if self.fPath == '': return
        mp = event.GetPosition()
        p_ = self.oData[self.fi]['tbPos']
        if p_ == (None,None) or p_ == ('D','D'): # position info is not determined or intentionally deleted 
            self.oData[self.fi]['tbPos'] = (mp[0], mp[1])
        else:
            r_ = (p_[0]-self.tagSz/2, p_[1]-self.tagSz/2, p_[0]+self.tagSz/2, p_[1]+self.tagSz/2) # x1,y1,x2,y2
            if r_[0] <= mp[0] <= r_[2] and r_[1] <= mp[1] <= r_[3]: # mouse clicked in the tag area
                self.oData[self.fi]['tbPos'] = ('D','D') # delete info
            else:
                self.oData[self.fi]['tbPos'] = (mp[0], mp[1])
        self.proc_img()

    #------------------------------------------------
    
    def onAdjustRect(self, event, flag):
        '''Adjusting p_rect of cv_proc, which defines 
        the rect (x1,y1,x2,y2) of bottom box paper panel
        '''
        if flag == 'm_left': self.cv_proc.p_rect[0] -= 1; self.cv_proc.p_rect[2] -= 1 # move left
        elif flag == 'm_right': self.cv_proc.p_rect[0] += 1; self.cv_proc.p_rect[2] += 1 # move right
        elif flag == 'm_up': self.cv_proc.p_rect[1] -= 1; self.cv_proc.p_rect[3] -= 1 # move up
        elif flag == 'm_down': self.cv_proc.p_rect[1] += 1; self.cv_proc.p_rect[3] += 1 # move down
        elif flag == 's_left': self.cv_proc.p_rect[2] -= 1 # shorten the width (actually moving x2)
        elif flag == 's_right': self.cv_proc.p_rect[2] += 1 # lengthen the width (moving x2)
        elif flag == 's_up': self.cv_proc.p_rect[3] -= 1 # shorten the height (moving y2)
        elif flag == 's_down': self.cv_proc.p_rect[3] += 1 # lengthen the height (moving y2)
        self.proc_img()
    
    #------------------------------------------------
    
    def proc_img(self):
        if self.fPath == '': return
        fp = path.join(self.fPath, 'f%06i.jpg'%self.fi)
        img = load_img(fp, flag='cv')
        rIMG, rTP, h2ac_dist = self.cv_proc.proc_img(img) # cv_proc.proc_img returns image, tag positions (head & tail-base), head-to-center distance
        self.loaded_img.SetBitmap( cvImg_to_wxBMP(rIMG) ) # display image
        #if rTP[0] != (-1, -1): # if it's not (-1,-1), update head tag position of the output data
        self.oData[self.fi]['hPos'] = rTP[0]
        self.oData[self.fi]['h2ac_dist'] = h2ac_dist
        #if rTP[1] != (-1, -1): # if it's not (-1,-1), update tail tag position of the output data
        self.oData[self.fi]['tbPos'] = rTP[1] 
        if self.is_running == True:
            if -1 not in [rTP[0][0], rTP[0][1], rTP[1][0], rTP[1][1]]: # all head and tail tag information were returned properly
                if self.fi < self.frame_cnt: # there's more frames to run
                    self.timer_run = wx.FutureCall(1, self.onRight, None, 'right') # keep continuous analysis

    #------------------------------------------------
    
    def onStartStopAnalyzeVideo(self, event):
        '''Choose a video file and starts analysis'''
        if self.session_start_time == -1: # not in analysis session. start a session
            dlg = wx.DirDialog(self, "Choose directory for analysis", getcwd(), wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_CANCEL: return
            self.fPath = dlg.GetPath()
            self.sString = self.fPath[-8:]
            self.frame_cnt = len(glob(path.join(self.fPath, '*.jpg')))
            if self.frame_cnt == 0:
                show_msg('No jpg frame images in the chosen directory.')
                self.fPath = ''
                return
            fNames = self.fPath.split('/')
            self.sTxt_fp.SetLabel( '%s / %s / %s / %s'%(fNames[-4],fNames[-3],fNames[-2],fNames[-1]) )
            result_csv_file = self.fPath + '.csv'
            if path.isfile(result_csv_file) == False: # result CSV file doesn't exist
                for i in range(1, self.frame_cnt+1):
                    self.oData[i] = dict( hPos = (None,None), tbPos = (None,None), h2ac_dist = None )
                    # hPos: head tag position, 
                    # tbPos: tail-base tag position, 
                    # h2ac_dist: distance from the head tag to the arena center 
            else: # result CSV file exists
                f = open(result_csv_file, 'r')
                lines = f.readlines()
                f.close()
                for i in range(1, self.frame_cnt+1):
                    self.oData[i] = dict( hPos = (None,None), tbPos = (None,None), h2ac_dist = None )
                    if i < len(lines):
                        items = [ x.strip() for x in lines[i].split(',') ]
                        idx_ = int(items[0])
                        if items[1] == 'None': hPos_val = (None, None)
                        elif items[1] == 'D': hPos_val = ('D', 'D')
                        else: hPos_val = ( int(items[1]), int(items[2]) )
                        if items[3] == 'None': tbPos_val = (None, None)
                        elif items[3] == 'D': tbPos_val = ('D', 'D')
                        else: tbPos_val = ( int(items[3]), int(items[4]) )
                        if items[7] == 'None': h2ac_dist_val = None
                        else: h2ac_dist_val = int(items[7])
                        self.oData[idx_]['hPos'] = copy(hPos_val)
                        self.oData[idx_]['tbPos'] = copy(tbPos_val)
                        self.oData[idx_]['h2ac_dist'] = copy(h2ac_dist_val)
            self.fi = 1
            self.session_start_time = time()
            self.btn_start.SetLabel('Stop analysis')
            ### start video recorder
            self.video_path = self.fPath + '.avi'
            fp = path.join(self.fPath, 'f%06i.jpg'%self.fi)
            img = load_img(fp, flag='cv')
            self.cv_proc.start_video_rec( self.video_path, img )
            self.proc_img() # process 1st image
        else: # in session. stop it.
            result = show_msg(msg='Save data?', cancel_btn = True)
            if result == True: self.onSave(None)
            if self.is_running == True: self.onSpace(None)
            self.session_start_time = -1
            self.sTxt_s_time.SetLabel('0:00:00')
            self.btn_start.SetLabel('Analyze video (folder)')
            self.sTxt_fp.SetLabel('')
            self.loaded_img.SetBitmap(wx.NullBitmap)
            self.fPath = ''
            self.sString = ''
            self.frame_cnt = 0
            self.fi = 0
            self.oData = {}
            self.cv_proc.stop_video_rec()
    
    # --------------------------------------------------       
    
    def onSave(self, event):
        WD = 0 # walking distance
        HM = 0 # head movements without walking (~ looking around)
        nfH = 0 # number of frames when only head tag is detected
        nfT = 0 # number of frames when only tail tag is detected
        nfB = 0 # number of frames when only both tags are detected
        nfN = 0 # number of frames when only no tags were detected 
        fp_ = self.fPath + '.csv'
        fh = open(fp_, 'w')
        fh.write('frame-index, hPosX, hPosY, tbPosX, tbPosY, WD, HM, h2ac_dist\n')
        hsf = self.vFPS/2 # half second frames
        for fi in range(1, self.frame_cnt+1):
            h_ = self.oData[fi]['hPos']
            t_ = self.oData[fi]['tbPos']
            if type(h_[0]) == int and type(t_[0]) == int: nfB += 1
            elif type(h_[0]) == int and type(t_[0]) != int: nfH += 1
            elif type(h_[0]) != int and type(t_[0]) == int: nfT += 1
            elif type(h_[0]) != int and type(t_[0]) != int: nfN += 1 
            wd_ = 0; hm_ = 0
            if fi > (hsf) and fi % hsf == 0:
                ### every half second, calculate walking-distance and head-movement
                ph_ = self.oData[fi-hsf]['hPos']
                pt_ = self.oData[fi-hsf]['tbPos']
                if type(h_[0])==int and type(h_[1])==int and type(ph_[0])==int and type(ph_[1])==int: # all the head position info available
                    hl = np.sqrt( (h_[0]-ph_[0])**2 + (h_[1]-ph_[1])**2 ) # line connecting two head tag positions
                    if hl > (self.tagSz/2): # if movement distance is too small, discard
                        if type(t_[0])==int and type(t_[1])==int and type(pt_[0])==int and type(pt_[1])==int: # all the tail position info available
                            hla = np.degrees(np.arctan2( (h_[1]-ph_[1]), (h_[0]-ph_[0]) )) # degree of line connecting two head tag positions 
                            tla = np.degrees(np.arctan2( (t_[1]-pt_[1]), (t_[0]-pt_[0]) )) # degree of line connecting two tail tag positions
                            angle_diff = calc_angle_diff(hla, tla)
                            if angle_diff < 45: # 45 degrees difference is considered as more or less similar direction to account it as 'walking'
                                wd_ = hl # walking distance for one frame
                                WD += hl # total walking distance
                        if wd_ == 0:
                            # if walking didn't happen, hl is recorded as head movement
                            hm_ = hl
                            HM += hl
            line = '%i, %s, %s, %s, %s, %i, %i, %s\n'%(fi, str(h_[0]), str(h_[1]), str(t_[0]), str(t_[1]), wd_, hm_, str(self.oData[fi]['h2ac_dist'])) 
            fh.write(line)
        fh.write('------------------------------------------------------------------\n')
        fh.write('Total walking distance, %i\n'%WD)
        fh.write('Total head movements without walking, %i\n'%HM)
        fh.write('------------------------------------------------------------------\n')
        fh.write('Number of frames when both tags are detected, %i\n'%nfB)
        fh.write('Number of frames when only head tag is detected, %i\n'%nfH)
        fh.write('Number of frames when only tail tag is detected, %i\n'%nfT)
        fh.write('Number of frames when no tags are detected, %i\n'%nfN)
        fh.close()

        msg = 'Saved.\n'
        chr_num = 50 # characters in one line
        if len(fp_) > chr_num:
            for i in range(len(fp_)/chr_num):
                msg += '%s\n'%(fp_[chr_num*i:chr_num*(i+1)])
            msg += '%s\n'%(fp_[chr_num*(i+1):])
        show_msg(msg, size=(400,200))

    # --------------------------------------------------       

    def onTimer(self, event):
        ''' Main timer 
        updating running time on the main window
        '''
        ### update several running time
        e_time = time() - self.program_start_time
        self.sTxt_pr_time.SetLabel( str(timedelta(seconds=e_time)).split('.')[0] )
        if self.session_start_time != -1:
            e_time = time() - self.session_start_time
            self.sTxt_s_time.SetLabel( str(timedelta(seconds=e_time)).split('.')[0] )

    # --------------------------------------------------

    def show_msg_in_statbar(self, msg, time=5000):
        self.SetStatusText(msg)
        wx.FutureCall(time, self.SetStatusText, "") # delete it after a while

    # --------------------------------------------------

    def onClose(self, event):
        self.timer.Stop()
        result = True
        if self.session_start_time != -1: # session is running
            result = show_msg(msg='Session is not stopped..\nUnsaved data will be lost. (Stop analysis or Cmd+S to save.)\nOkay to proceed to exit?', cancel_btn = True)
        if result == True:
            if self.cv_proc.video_rec != None: self.cv_proc.stop_video_rec()
            wx.FutureCall(500, self.Destroy)

# ======================================================

class AMAApp(wx.App):
    def OnInit(self):
        self.frame = AMAFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

# ======================================================

if __name__ == '__main__':
    if len(argv) > 1:
        if argv[1] == '-w': GNU_notice(1)
        elif argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        CWD = getcwd()
        app = AMAApp(redirect = False)
        app.MainLoop()




