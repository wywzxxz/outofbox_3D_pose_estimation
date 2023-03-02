import sys, os 
sys.path.append(os.path.abspath( __file__+"/../../models/MHFormer" ))
sys.path.append(os.path.abspath( __file__+"/../../models/MHFormer/demo" ))
import vis
from vis import get_pose2D, Model, normalize_screen_coordinates, camera_to_world, show2Dpose, showimage, show3Dpose, img2video

import matplotlib
import matplotlib.pyplot as plt 
import glob
import argparse
import torch, numpy as np
import cv2
from tqdm import tqdm
import copy
import matplotlib.gridspec as gridspec
import json

def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']    
    cap = cv2.VideoCapture(video_path)
    video_length = keypoints.shape[1]#int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("keypoints",keypoints.shape)
    ## 3D
    result_3D=[]
    print('\nGenerating 3D pose... video_length=',video_length)
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        img_size = img.shape

        ## input frames
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        
        output_3D_non_flip = model(input_2D[:, 0])  # [1, 351, 17, 2] -> ([1, 351, 17, 3]
        output_3D_flip     = model(input_2D[:, 1])
        print(">>>>",input_2D[:, 0].shape)        

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0:, args.pad].unsqueeze(1)         
        #output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        result_3D.append(post_out.tolist())
        
        post_out[:, 2] -= np.min(post_out[:, 2])

        input_2D_no = input_2D_no[args.pad]

        ## 2D
        image = show2Dpose(input_2D_no, copy.deepcopy(img))

        output_dir_2D = output_dir +'pose2D\\'
        os.makedirs(output_dir_2D, exist_ok=True)
        if not cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image):
            raise Exception("cv2.imwrite return False")

        ## 3D
        fig = plt.figure( figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose( post_out, ax)

        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.close()        
    print("result_3D",len(result_3D))
    with open(output_dir+"/info_3d.json","w") as f:
        json.dump({"keypoint":result_3D},f)
    print('Generating 3D pose successful!')
    ## all
    image_dir = 'results/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))
    
    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close()




if __name__ == "__main__":        
    video_name = sys.argv[1] if len(sys.argv)>1 else ""
    sys.argv=sys.argv[:1]
    if video_name=="":
        video_name="sample"
    video_path = os.path.abspath(__file__+"/../../videos/"+video_name+"/raw.mp4")
    output_dir = os.path.abspath(__file__+"/../../videos/"+video_name)+"\\"

    cwd=os.getcwd()
    os.chdir(os.path.abspath(__file__+"/../../models/MHFormer"))
    
    get_pose2D(video_path, output_dir)     
    get_pose3D(video_path, output_dir)
    vis.video_name = "output"
    img2video(video_path, output_dir)
    os.chdir(cwd)
    print('Generating demo successful!')
