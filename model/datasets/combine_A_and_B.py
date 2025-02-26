# “스케치(폴더 A) + 실제 이미지(폴더 B)”를 쌍으로 매칭하여 (A|B) 형식으로 합치는 과정을 수행
import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool, freeze_support

# 가능한 확장자 목록
VALID_EXTS = ['.png', '.jpg', '.jpeg']

def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    if im_A is None or im_B is None:
        print(f"Warning: Failed to read {path_A} or {path_B}. Skipping.")
        return
    # 가로 방향으로 이어붙이기
    im_AB = np.concatenate([im_A, im_B], axis=1)
    cv2.imwrite(path_AB, im_AB)

if __name__ == '__main__':
    freeze_support()  # Windows 환경에서 필요할 수 있음

    parser = argparse.ArgumentParser('create image pairs')
    parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
    parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
    parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
    parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
    parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) -> (0001_AB)', action='store_true')
    parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If true, single CPU execution', action='store_true', default=False)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))

    # fold_A 내부 폴더 목록 (예: train, test, val)
    splits = os.listdir(args.fold_A)

    if not args.no_multiprocessing:
        pool = Pool()

    for sp in splits:
        img_fold_A = os.path.join(args.fold_A, sp)
        img_fold_B = os.path.join(args.fold_B, sp)
        if not os.path.isdir(img_fold_A):
            continue

        img_list = os.listdir(img_fold_A)
        if args.use_AB:
            img_list = [img_path for img_path in img_list if '_A.' in img_path]

        num_imgs = min(args.num_imgs, len(img_list))
        print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))

        img_fold_AB = os.path.join(args.fold_AB, sp)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)
        print('split = %s, number of images = %d' % (sp, num_imgs))

        for n in range(num_imgs):
            name_A = img_list[n]
            path_A = os.path.join(img_fold_A, name_A)

            # use_AB 옵션 처리
            if args.use_AB:
                name_B = name_A.replace('_A.', '_B.')
            else:
                # 기존: 이름(B) = A파일 이름에서 확장자 제거
                # 수정: 파일 이름에서 공통 접두어(예: "fimage0000")만 추출하여 B 파일을 찾는다.
                base_A = os.path.splitext(name_A)[0]  # 예: "fimage0000_drawn"
                # 여기서 "_"를 기준으로 분할하여 첫 번째 부분을 사용
                common_prefix = base_A.split('_')[0]   # 예: "fimage0000"
                
                # B 폴더에서 파일 이름이 common_prefix로 시작하는 파일 찾기
                found_path_B = None
                for file in os.listdir(img_fold_B):
                    file_base = os.path.splitext(file)[0]
                    if file_base.startswith(common_prefix):
                        # 확장자가 VALID_EXTS에 포함되어 있는지 확인
                        ext = os.path.splitext(file)[1].lower()
                        if ext in VALID_EXTS:
                            found_path_B = os.path.join(img_fold_B, file)
                            break
                if found_path_B is None:
                    # 매칭되는 파일 없음 → skip
                    print(f"Warning: No matching file for {name_A} in {img_fold_B}")
                    continue

                name_B = os.path.basename(found_path_B)

            # 만약 args.use_AB가 True면, name_B은 이미 설정됨.
            if not args.use_AB and found_path_B is not None:
                # found_path_B는 위에서 결정됨.
                pass

            # 최종 출력 파일 이름 결정: A 파일 이름 그대로 사용
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')
            path_AB = os.path.join(img_fold_AB, name_AB)

            if not args.no_multiprocessing:
                if args.use_AB:
                    pool.apply_async(image_write, args=(path_A, os.path.join(img_fold_B, name_B), path_AB))
                else:
                    pool.apply_async(image_write, args=(path_A, found_path_B, path_AB))
            else:
                if args.use_AB:
                    path_B_final = os.path.join(img_fold_B, name_B)
                else:
                    path_B_final = found_path_B
                im_A = cv2.imread(path_A, 1)
                im_B = cv2.imread(path_B_final, 1)
                if im_A is None or im_B is None:
                    print(f"Warning: Failed to read {path_A} or {path_B_final}. Skipping.")
                    continue
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)

    if not args.no_multiprocessing:
        pool.close()
        pool.join()
