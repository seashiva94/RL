import pygame
import random
import numpy as np
import tensorflow as tf
import cv2

FPS = 60
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 10

BALL_WIDTH = 10
BALL_HEIGHT = 10

PADDLE_SPEED = 2
BALL_X_SPEED = 3
BALL_Y_SPEED = 2

WHITE = (255,255,255)
BLACK = (0,0,0)

screen= pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

def drawBall(ballx, bally):
    ball = pygame.Rect(ballx, bally, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)

def drawPaddle1(paddle1y):
    paddle1 = pygame.Rect(PADDLE_BUFFER, paddle1y, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle1)

def drawPaddle2(paddle2y):
    paddle2 = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2y, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle2)

def updateBall(paddle1y, paddle2y, ballx, bally, ballxdir, ballydir):
    # updatte x and y pos of ball

    ballx = ballx + ballxdir * BALL_X_SPEED
    bally = bally + ballydir * BALL_Y_SPEED
    score = 0

    # check colissons for change direction
    # if ball hits left side switch direction
    if(ballx <= PADDLE_BUFFER + PADDLE_WIDTH and bally + BALL_HEIGHT >= paddle1y and bally - BALL_HEIGHT <= paddle1y + PADDLE_HEIGHT):
        ballxdir = 1
    elif(ballx <=0):
        ballxdir = 1
        score = -1
        return [score, paddle1y, paddle2y, ballx, bally, ballxdir, ballydir]


    if(ballx >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and bally + BALL_HEIGHT >= paddle2y and bally - BALL_HEIGHT <=paddle2y + PADDLE_HEIGHT):
        ballxdir = -1

    elif(ballx >= WINDOW_WIDTH - BALL_WIDTH):
        ballxdir = -1
        score = 1
        return [score, paddle1y, paddle2y, ballx, bally, ballxdir, ballydir]

    if(bally <=0):
        bally = 0
        ballydir = 1
    elif(bally >= WINDOW_HEIGHT - BALL_HEIGHT):
        bally = WINDOW_HEIGHT - BALL_HEIGHT
        ballydir = -1

    return [score, paddle1y, paddle2y, ballx, bally, ballxdir, ballydir]


def  updatePaddle1(action, paddle1y):

    #move up
    if(action[1] == 1):
        paddle1y = paddle1y - PADDLE_SPEED

    # move_down
    if(action[2] == 1):
        paddle1y = paddle1y + PADDLE_SPEED

    # keep in screen
    if(paddle1y <0):
        paddle1y = 0
    if(paddle1y > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1y = WINDOW_HEIGHT - PADDLE_HEIGHT

    return paddle1y


def  updatePaddle2(paddle2y, bally):

    if(paddle2y + PADDLE_HEIGHT/2 < bally + BALL_HEIGHT/2):
        paddle2y = paddle2y + PADDLE_SPEED

    if(paddle2y + PADDLE_HEIGHT/2 > bally + BALL_HEIGHT/2):
        paddle2y = paddle2y - PADDLE_SPEED

    if(paddle2y <0):
        paddle1y = 0
    if(paddle2y > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1y = WINDOW_HEIGHT - PADDLE_HEIGHT

    return paddle2y


class Pong:
    def __init__(self):
       
        #score variable
        self.tally = 0
        # paddle positions in middle initially
        self.paddle1y = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2
        self.paddle2y = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2
        # balld idectoijn
        self.ballxdir = 1
        self.ballydir = 1
        self.ballx = WINDOW_HEIGHT/2 - PADDLE_HEIGHT/2

        #randomly initialize ball
        num = random.randint(0,9)
        if(0 < num < 3):
            self.ballxdir = 1
            self.ballydir = 1
        if(3 <= num < 5):
            self.ballxdir = -1
            self.ballydir = 1
        if(5 <= num < 8):
            self.ballxdir = 1
            self.ballydir = -1
        if(8 <= num <= 10):
            self.ballxdir = -1
            self.ballydir = -1
            
        num = random.randint(0,9)
        self.bally = num*(WINDOW_HEIGHT - PADDLE_HEIGHT)/9

    def getPresentFrame(self):
        # eveny queue
        pygame.event.pump()
        #draw a new screen 
        screen.fill(BLACK)
        drawPaddle1(self.paddle1y)
        drawPaddle2(self.paddle2y)
        drawBall(self.ballx, self.bally)

        # get pixels of the image
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # update window
        pygame.display.flip()

        return image_data
    
    def getNextFrame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        self.paddle1y = updatePaddle1(action, self.paddle1y)
        drawPaddle1(self.paddle1y)

        self.paddle2y = updatePaddle2( self.paddle2y, self.bally)
        drawPaddle2(self.paddle2y)

        [score, self.paddle1y, self.paddle2y, self.ballx, self.bally, self.ballxdir, self.ballydir] = updateBall(self.paddle1y, self.paddle2y, self.ballx, self.bally, self.ballxdir, self.ballydir)
        
        drawBall(self.ballx, self.bally)
        # get pixels of the image
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # update window
        pygame.display.flip()

        self.tally = self.tally + score
        return [score, image_data]
    
