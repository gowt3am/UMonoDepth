import pangolin, yaml, time
import numpy as np
import OpenGL.GL as gl

def main(trajectory_file='outputs/outdoor/50all/run1_trajectory.txt', settings_file='data/outdoors/settings.yaml'):
    with open(trajectory_file, 'r') as f:
        poses = [line.strip().split(' ') for line in f.readlines()]
    n = len(poses)
    print('Total found poses = {}'.format(n))
    poses = np.array(poses).astype(np.float32).reshape(n, 12)
    with open(settings_file) as f:                                # Default values for settings.yaml given below in comments
        settings = yaml.load(f, Loader=yaml.FullLoader)
    DrawTrajectory(poses, settings)

def DrawTrajectory(poses, s):
    width = s["Camera.width"]         # 640
    height = s["Camera.height"]       # 480
    viewX = s["Viewer.ViewpointX"]    # 0
    viewY = s["Viewer.ViewpointY"]    # -0.7
    viewZ = s["Viewer.ViewpointZ"]    # -1.8
    viewF = s["Viewer.ViewpointF"]    # 500
    w = s["Viewer.CameraSize"]        # 0.08
    h = w*0.75
    z = w*0.6
    
    pangolin.CreateWindowAndBind('Trajectory Viewer', width, height)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    
    scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(width, height, viewF, viewF, 512, 389, 0.1, 1000),
            pangolin.ModelViewLookAt(viewX, viewY, viewZ, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -width/height)
    dcam.SetHandler(handler)
    
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
      
        for i in range(len(poses)):
            if i % 20 == 0:
                pose = np.reshape(poses[i], (3, 4))
                '''
                pose = np.vstack([pose, np.array([0, 0, 0, 1])])
                pose = np.linalg.inv(pose)[:3, :] 
                '''
                gl.glLineWidth(2)
                gl.glColor3f(0.3, 0.7, 0.5)
                pangolin.DrawCamera(np.vstack([pose, np.array([0, 0, 0, 1])]), w, h, z)
                
                O = pose[:, 3]
                X = O + w * pose[:, 0]
                Y = O + w * pose[:, 1]
                Z = O + w * pose[:, 2]
                gl.glBegin(gl.GL_LINES)
                gl.glColor3f(1.0, 0.0, 0.0)     # R: Left/Right
                gl.glVertex3d(O[0], O[1], O[2])
                gl.glVertex3d(X[0], X[1], X[2])
                gl.glColor3f(0.0, 1.0, 0.0)     # G: Above/Below
                gl.glVertex3d(O[0], O[1], O[2])
                gl.glVertex3d(Y[0], Y[1], Y[2])
                gl.glColor3f(0.0, 0.0, 1.0)     # B: Front/Back
                gl.glVertex3d(O[0], O[1], O[2])
                gl.glVertex3d(Z[0], Z[1], Z[2])
                gl.glEnd()

        for i in range(len(poses)-1):
            gl.glLineWidth(3)
            gl.glColor3f(0.0, 0.0, 0.0)
            gl.glBegin(gl.GL_LINES)
            '''
            p1 = np.reshape(poses[i], (3, 4))
            p1 = np.vstack([p1, np.array([0, 0, 0, 1])])
            p1 = np.linalg.inv(p1)[:3, 3]
            p2 = np.reshape(poses[i+1], (3, 4))
            p2 = np.vstack([p2, np.array([0, 0, 0, 1])])
            p2 = np.linalg.inv(p2)[:3, 3]
            '''
            p1 = np.reshape(poses[i], (3, 4))[:, 3]
            p2 = np.reshape(poses[i+1], (3, 4))[:, 3]
            
            gl.glVertex3d(p1[0], p1[1], p1[2])
            gl.glVertex3d(p2[0], p2[1], p2[2])
            gl.glEnd()
        pangolin.FinishFrame()
        
if __name__ == '__main__':
    main()