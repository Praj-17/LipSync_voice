import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import argparse  # Changed: Import argparse to create args-like object

sys.path.insert(0, 'video_retalking')
sys.path.insert(0, 'video_retalking/third_part')
sys.path.insert(0, 'video_retalking/third_part/GPEN')
sys.path.insert(0, 'video_retalking/third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")

class LipSyncer:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.enhancer = FaceEnhancement(base_dir='video_retalking/checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                                sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=self.device)
        self.restorer = GFPGANer(model_path='video_retalking/checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
                            channel_multiplier=2, bg_upsampler=None)
        self.croper = Croper('video_retalking/checkpoints/shape_predictor_68_face_landmarks.dat')

    def inference(self, face_path, audio_path):
        return self.run_inference(face=face_path, audio_path=audio_path)


    def run_inference(
        self, 
        face,                # Replaces args.face
        audio_path,          # Replaces args.audio            # Replaces args.outfile
        tmp_dir='temp',      # Replaces args.tmp_dir with default value
        fps=None,            # Replaces args.fps
        crop=(0, -1, 0, -1), # Replaces args.crop with default value
        re_preprocess=False, # Replaces args.re_preprocess
        face3d_net_path='video_retalking/checkpoints/face3d_pretrain_epoch_20.pth',  # Replaces args.face3d_net_path with default path
        LNet_path = 'video_retalking/checkpoints/LNet.pth',
        ENet_path = 'video_retalking/checkpoints/ENet.pth',
        exp_img=None,        # Replaces args.exp_img
        LNet_batch_size=8,   # Replaces args.LNet_batch_size with default value
        img_size=256,        # Replaces args.img_size with default value
        up_face='original',  # Replaces args.up_face with default value
        without_rl1=False,   # Replaces args.without_rl1
        DNet_path='video_retalking/checkpoints/DNet.pt',
        nosmooth = False,  # Added: Replaces args.DNet_path with default path
        pads=[0, 20, 0, 0],  # Added: Replaces args.DNet_path with default path
        face_det_batch_size = 4
    ):
        outfile = os.path.join(os.path.dirname(face), "result.mp4")
        # Changed: Create an args-like object using argparse.Namespace
        args = argparse.Namespace(
            face=face,
            audio=audio_path,
            outfile=outfile,
            tmp_dir=tmp_dir,
            fps=fps,
            crop=crop,
            re_preprocess=re_preprocess,
            face3d_net_path=face3d_net_path,
            exp_img=exp_img,
            LNet_batch_size=LNet_batch_size,
            LNet_path=LNet_path,
            ENet_path=ENet_path,
            img_size=img_size,
            up_face=up_face,
            without_rl1=without_rl1,
            nosmooth = nosmooth,
            pads=pads,
            face_det_batch_size=face_det_batch_size,
            DNet_path=DNet_path  # Added: Include DNet_path in args
        )
        
        print("Inferenceing on ", args.face, args.audio)

        print('[Info] Using {} for inference.'.format(self.device))
        os.makedirs(os.path.join(tmp_dir), exist_ok=True)  # Changed: Use tmp_dir directly


        base_name = os.path.basename(args.face)  # Changed: Use os.path.basename
        static = False
        if os.path.isfile(args.face) and os.path.splitext(args.face)[1].lower() in ['.jpg', '.png', '.jpeg']:  # Changed: Use os.path.splitext and lower()
            static = True
        if not os.path.isfile(args.face):
            raise ValueError('--face argument must be a valid path to video/image file')
        elif os.path.splitext(args.face)[1].lower() in ['.jpg', '.png', '.jpeg']:
            full_frames = [cv2.imread(args.face)]
            actual_fps = args.fps if args.fps is not None else 25  # Changed: Default FPS if not provided
        else:
            video_stream = cv2.VideoCapture(args.face)
            video_fps = video_stream.get(cv2.CAP_PROP_FPS)
            actual_fps = args.fps if args.fps is not None else video_fps  # Changed: Handle fps parameter

            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                y1, y2, x1, x2 = args.crop  # Changed: Use args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        print("[Step 0] Number of frames available for inference: " + str(len(full_frames)))
        # Face detection & cropping, cropping the first frame as the style of FFHQ
        
        full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
        full_frames_RGB, crop_coords, quad = self.croper.crop(full_frames_RGB, xsize=512)

        clx, cly, crx, cry = crop_coords
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly + ly, min(cly + ry, full_frames[0].shape[0]), clx + lx, min(clx + rx, full_frames[0].shape[1])
        # original_size = (ox2 - ox1, oy2 - oy1)
        frames_pil = [Image.fromarray(cv2.resize(frame, (256, 256))) for frame in full_frames_RGB]

        # Get the landmarks according to the detected face.
        landmarks_path = os.path.join(args.tmp_dir, f"{base_name}_landmarks.txt")  # Changed: Construct landmarks path
        if not os.path.isfile(landmarks_path) or args.re_preprocess:  # Changed: Use args.re_preprocess
            print('[Step 1] Landmarks Extraction in Video.')
            kp_extractor = KeypointExtractor()
            lm = kp_extractor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print('[Step 1] Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(full_frames), -1, 2])

        coeffs_path = os.path.join(args.tmp_dir, f"{base_name}_coeffs.npy")  # Changed: Construct coeffs path
        if not os.path.isfile(coeffs_path) or args.exp_img is not None or args.re_preprocess:  # Changed: Use args.exp_img and args.re_preprocess
            net_recon = load_face3d_net(args.face3d_net_path, self.device)  # Changed: Use args.face3d_net_path
            lm3d_std = load_lm3d('video_retalking/checkpoints/BFM')

            video_coeffs = []
            for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
                frame = frames_pil[idx]
                W, H = frame.size
                lm_idx = lm[idx].reshape([-1, 2])
                if np.mean(lm_idx) == -1:
                    lm_idx = (lm3d_std[:, :2] + 1) / 2.
                    lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
                else:
                    lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

                trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_idx_tensor = torch.tensor(np.array(im_idx) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    coeffs = split_coeff(net_recon(im_idx_tensor))

                pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
                pred_coeff = np.concatenate([
                    pred_coeff['id'],
                    pred_coeff['exp'],
                    pred_coeff['tex'],
                    pred_coeff['angle'],
                    pred_coeff['gamma'],
                    pred_coeff['trans'],
                    trans_params[None]
                ], 1)
                video_coeffs.append(pred_coeff)
            semantic_npy = np.array(video_coeffs)[:, 0]
            np.save(coeffs_path, semantic_npy)
        else:
            print('[Step 2] Using saved coeffs.')
            semantic_npy = np.load(coeffs_path).astype(np.float32)

        # Generate the 3dmm coeff from a single image
        if args.exp_img is not None and (args.exp_img.lower().endswith('.png') or args.exp_img.lower().endswith('.jpg')):  # Changed: Case-insensitive check
            print('Extracting expression from', args.exp_img)
            exp_pil = Image.open(args.exp_img).convert('RGB')
            lm3d_std = load_lm3d('third_part/face3d/BFM')

            W, H = exp_pil.size
            kp_extractor = KeypointExtractor()
            temp_landmarks_path = os.path.join(args.tmp_dir, f"{base_name}_temp.txt")  # Changed: Temporary landmarks path
            lm_exp = kp_extractor.extract_keypoint([exp_pil], temp_landmarks_path)[0]
            if np.mean(lm_exp) == -1:
                lm_exp = (lm3d_std[:, :2] + 1) / 2.
                lm_exp = np.concatenate([lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
            else:
                lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

            trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_exp_tensor = torch.tensor(np.array(im_exp) / 255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
            with torch.no_grad():
                expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
            del net_recon
        elif args.exp_img == 'smile':
            expression = torch.tensor(loadmat('video_retalking/checkpoints/expression.mat')['expression_mouth'])[0]
        else:
            print('Using expression center')
            expression = torch.tensor(loadmat('video_retalking/checkpoints/expression.mat')['expression_center'])[0]

        # Load DNet, model (LNet and ENet)
        D_Net, model = load_model(args, self.device)  # Changed: Pass args to load_model

        stabilized_path = os.path.join(args.tmp_dir, f"{base_name}_stablized.npy")  # Changed: Stabilized path
        if not os.path.isfile(stabilized_path) or args.re_preprocess:  # Changed: Use args.re_preprocess
            imgs = []
            for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):
                if args.up_face == 'original':  # Changed: Use args.up_face
                    source_img = trans_image(frames_pil[0]).unsqueeze(0).to(self.device)
                    semantic_source_numpy = semantic_npy[0:1]
                else:
                    source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(self.device)
                    semantic_source_numpy = semantic_npy[idx:idx + 1]
                ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
                coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(self.device)

                # Hacking the new expression
                coeff[:, :64, :] = expression[None, :64, None].to(self.device)
                with torch.no_grad():
                    output = D_Net(source_img, coeff)
                img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1, 2, 0).cpu().clamp_(-1, 1).numpy() + 1) / 2. * 255)
                imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
            np.save(stabilized_path, imgs)
            del D_Net
        else:
            print('[Step 3] Using saved stabilized video.')
            imgs = np.load(stabilized_path)
        torch.cuda.empty_cache()

        # Handle audio processing
        if not args.audio.endswith('.wav'):
            temp_wav = os.path.join(args.tmp_dir, 'temp.wav')  # Changed: Temporary WAV path
            command = f'ffmpeg -loglevel error -y -i {args.audio} -strict -2 {temp_wav}'
            subprocess.call(command, shell=True)
            args.audio = temp_wav
        wav = audio.load_wav(args.audio, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80. / actual_fps, 0, []
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1

        print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
        imgs = imgs[:len(mel_chunks)]
        full_frames = full_frames[:len(mel_chunks)]
        lm = lm[:len(mel_chunks)]

        imgs_enhanced = []
        for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
            img = imgs[idx]
            pred, _, _ = self.enhancer.process(img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        
        gen = self.datagen(
            frames=imgs_enhanced.copy(),
            mels=mel_chunks,
            full_frames=full_frames,
            frames_pil=None,
            cox=(oy1, oy2, ox1, ox2),
            args=args  # Changed: Pass args to datagen
        )

        frame_h, frame_w = full_frames[0].shape[:-1]
        output_path = os.path.join(args.tmp_dir, 'result.mp4')  # Changed: Output path
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), actual_fps, (frame_w, frame_h))  # Changed: Use actual_fps

        if args.up_face != 'original':
            instance = GANimationModel()
            instance.initialize()
            instance.setup()

        kp_extractor = KeypointExtractor()
        for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(self.device) / 255.  # BGR -> RGB

            with torch.no_grad():
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                pred, low_res = model(mel_batch, img_batch, reference)
                pred = torch.clamp(pred, 0, 1)

                if args.up_face in ['sad', 'angry', 'surprise']:
                    tar_aus = exp_aus_dict[args.up_face]
                else:
                    tar_aus = None  # Ensure tar_aus is defined

                if args.up_face == 'original':
                    cur_gen_faces = img_original
                else:
                    test_batch = {
                        'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'),
                        'tar_aus': tar_aus.repeat(len(incomplete), 1)
                    }
                    instance.feed_batch(test_batch)
                    instance.forward()
                    cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')

                if args.without_rl1 is not False:
                    incomplete, reference = torch.split(img_batch, 3, dim=1)
                    mask = torch.where(incomplete == 0, torch.ones_like(incomplete), torch.zeros_like(incomplete))
                    pred = pred * mask + cur_gen_faces * (1 - mask)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            torch.cuda.empty_cache()
            for p, f, xf, c in zip(pred, frames, f_frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                ff = xf.copy()
                ff[y1:y2, x1:x2] = p

                # Mouth region enhancement by GFPGAN
                cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                    ff, has_aligned=False, only_center_face=True, paste_back=True)
                # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
                mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                mouth_mask = np.zeros_like(restored_img)  # Changed: Renamed to mouth_mask for clarity
                tmp_mask = self.enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
                mouth_mask[y1:y2, x1:x2] = cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

                height, width = ff.shape[:2]
                restored_img_resized, ff_resized, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouth_mask))]
                img = Laplacian_Pyramid_Blending_with_mask(restored_img_resized, ff_resized, full_mask[:, :, 0], 10)
                pp = np.uint8(cv2.resize(np.clip(img, 0, 255), (width, height)))

                pp, orig_faces, enhanced_faces = self.enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
                out.write(pp)
        out.release()

        # Save the output file
        if not os.path.isdir(os.path.dirname(args.outfile)):
            os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        command = f'ffmpeg -loglevel error -y -i {args.audio} -i {output_path} -strict -2 -q:v 1 {args.outfile}'  # Changed: Use args.outfile and paths
        subprocess.call(command, shell=platform.system() != 'Windows')
        print('outfile:', args.outfile)
        return outfile

    # Corrected datagen function to fix the 'shape' attribute error and variable resetting
    def datagen(self, frames, mels, full_frames, frames_pil, cox, args):  # Changed: Accept args as a parameter
        img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch, img_original = [], [], [], [], [], [], []
        base_name = os.path.basename(args.face)  # Changed: Use args.face to get base_name
        refs = []
        image_size = args.img_size  # Changed: Use args.img_size

        # Original frames
        kp_extractor = KeypointExtractor()
        fr_pil = [Image.fromarray(frame) for frame in frames]
        landmarks_file = os.path.join(args.tmp_dir, f"{base_name}x12_landmarks.txt")  # Changed: Construct landmarks path
        lms = kp_extractor.extract_keypoint(fr_pil, landmarks_file)  # Changed: Use landmarks_file
        frames_pil = [(lm, frame) for frame, lm in zip(fr_pil, lms)]  # frames is the cropped version of modified face
        crops, orig_images, quads = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
        inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
        del kp_extractor.detector

        oy1, oy2, ox1, ox2 = cox
        face_det_results = face_detect(full_frames, args, jaw_correction=True)  # Changed: Pass args to face_detect

        for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
            imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
                cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))  # Changed: Use args.crop coords

            ff = full_frame.copy()
            ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
            oface, coords = face_det
            y1, y2, x1, x2 = coords
            refs.append(ff[y1: y2, x1:x2])

        for i, m in enumerate(mels):
            # Changed: Use static condition based on file type
            idx = 0 if os.path.splitext(args.face)[1].lower() in ['.jpg', '.png', '.jpeg'] else i % len(frames)
            frame_to_save = frames[idx].copy()
            face = refs[idx]
            oface, coords = face_det_results[idx]

            face = cv2.resize(face, (args.img_size, args.img_size))  # Changed: Use args.img_size
            oface = cv2.resize(oface, (args.img_size, args.img_size))  # Changed: Use args.img_size

            img_batch.append(oface)
            ref_batch.append(face)
            mel_batch.append(m)
            coords_batch.append(coords)
            frame_batch.append(frame_to_save)
            full_frame_batch.append(full_frames[idx].copy())

            if len(img_batch) >= args.LNet_batch_size:  # Changed: Use args.LNet_batch_size
                img_batch_np, mel_batch_np, ref_batch_np = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
                img_masked = img_batch_np.copy()
                img_original = img_batch_np.copy()
                img_masked[:, args.img_size // 2:] = 0  # Changed: Use args.img_size
                img_batch_np = np.concatenate((img_masked, ref_batch_np), axis=3) / 255.
                # Fixed: Use mels[0].shape instead of mels.shape
                mel_batch_np = np.reshape(mel_batch, [len(mel_batch), mels[0].shape[0], mels[0].shape[1], 1])

                yield img_batch_np, mel_batch_np, frame_batch, coords_batch, img_original, full_frame_batch
                img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch, img_original = [], [], [], [], [], [], []

        if len(img_batch) > 0:
            img_batch_np, mel_batch_np, ref_batch_np = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch_np.copy()
            img_original = img_batch_np.copy()
            img_masked[:, args.img_size // 2:] = 0  # Changed: Use args.img_size
            img_batch_np = np.concatenate((img_masked, ref_batch_np), axis=3) / 255.
            # Fixed: Use mels[0].shape instead of mels.shape
            mel_batch_np = np.reshape(mel_batch, [len(mel_batch), mels[0].shape[0], mels[0].shape[1], 1])
            yield img_batch_np, mel_batch_np, frame_batch, coords_batch, img_original, full_frame_batch


# Removed the CLI entry point
# if __name__ == '__main__':
#     main()
# Changed: Removed CLI entry point, use run_inference function instead

if __name__ == '__main__':

    Ls = LipSyncer()

    # Example usage:
    Ls.inference(
        face='examples/face/7.mp4',
        audio_path='examples/audio/output_waved_class.wav',
        outfile='results/obama_7.mp4',
    )
