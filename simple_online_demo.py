import os
import pdb
import pickle
import time

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from torch.autograd import Variable

from mean import get_mean, get_std
from model import generate_model
from opts import parse_opts_online
from spatial_transforms import *


def load_models(opt):
    opt.resume_path = opt.resume_path
    opt.pretrain_path = opt.pretrain_path
    opt.sample_duration = opt.sample_duration
    opt.model = opt.model
    opt.model_depth = opt.model_depth
    opt.width_mult = opt.width_mult
    opt.modality = opt.modality
    opt.resnet_shortcut = opt.resnet_shortcut
    opt.n_classes = opt.n_classes
    opt.n_finetune_classes = opt.n_finetune_classes

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location=torch.device('cpu'))
        pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        classifier.load_state_dict(pretrained_dict, strict=False)

    print('Model \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return classifier


def calculate_centroids(X_train, y_train):
    clf = NearestCentroid()
    clf.fit(X_train, y_train)

    return clf.centroids_, clf.classes_


def calculate_new_gesture(frame_buffer):
    clip = []
    for frame in frame_buffer[-32:]:
        _frame = cv2.resize(frame, (320, 240))
        _frame = Image.fromarray(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))
        _frame = _frame.convert('RGB')
        _frame = spatial_transform(_frame)
        clip.append(_frame)

    im_dim = clip[0].size()[-2:]
    try:
        test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
    except Exception as e:
        pdb.set_trace()
        raise e
    inputs = torch.cat([test_data], 0).view(1, 3, opt.sample_duration, 112, 112)

    with torch.no_grad():
        inputs = Variable(inputs)
        inputs_clf = inputs[:, :, :, :, :]
        inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::1, :, :])

        outputs_clf = feature_extractor(inputs_clf)

    return outputs_clf


if __name__ == '__main__':
    with open('results/train_resnext_with_class.dump', 'rb') as f:
        train_embs = pickle.load(f)

    X_train = []
    y_train = []
    for x in train_embs:
        X_train.append(train_embs[x]['emb'])
        y_train.append(train_embs[x]['y'])

    centroids, classes = calculate_centroids(X_train, y_train)

    neigh_clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
    neigh_clf.fit(centroids, classes)

    opt = parse_opts_online()

    feature_extractor = load_models(opt)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value), norm_method
    ])

    feature_extractor.eval()
    spatial_transform.randomize_parameters()

    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    t1 = time.time()
    frame_buffer = []
    new_gesture_buffer = []
    number_of_new_gesture_record = 4
    current_gesture_record = 0

    num_frame = 0
    class_validation = None
    diff = None
    compute_diff = False
    while cap.isOpened():
        ret, frame = cap.read()

        t2 = time.time()
        delta = t2 - t1

        if current_gesture_record <= number_of_new_gesture_record:
            if delta < 3:
                text = 'Wait for {} to record new gesture'.format(str(round(3 - delta, 1)))
                cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

            elif delta >= 3 and delta < 5:
                text = 'Recording gesture'
                cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

                frame_buffer.append(frame)
            else:
                new_gesture = calculate_new_gesture(frame_buffer).cpu().detach().numpy().reshape(-1)
                new_gesture_buffer.append(new_gesture)

                frame_buffer = []
                current_gesture_record += 1

                if current_gesture_record == number_of_new_gesture_record:
                    X_train.extend(new_gesture_buffer)
                    y_train.extend(['new_gesture' for x in range(number_of_new_gesture_record)])

                    centroids, classes = calculate_centroids(X_train, y_train)

                    neigh_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
                    neigh_clf.fit(centroids, classes)

                t1 = time.time()


        else:
            if delta < 3:
                text = 'Wait for {} to record validation gesture'.format(str(round(3 - delta, 1)))
                cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

            elif delta >= 3 and delta < 5:
                text = 'Recording gesture'
                cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

                frame_buffer.append(frame)

            elif delta >= 5 and delta < 8:
                if compute_diff == False:
                    compute_diff = True

                    gesture = calculate_new_gesture(frame_buffer)

                    outputs_clf = neigh_clf.predict_proba(gesture).reshape(-1)

                    best2, best1 = tuple(outputs_clf.argsort()[-2:][::1])

                    diff = outputs_clf[best1] - outputs_clf[best2]
                    if diff > opt.clf_threshold_final:
                        class_validation = neigh_clf.classes_[best1]

                    print(neigh_clf.classes_[best1])
                    print(outputs_clf[best1])

                    print(neigh_clf.classes_[best2])
                    print(outputs_clf[best2])

                    print()


                else:
                    # text = 'Validation gesture is: {}, diff: {}'.format(class_validation, diff)
                    text = 'Validation gesture is: {}'.format(class_validation)
                    cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('frame', frame)

            else:
                compute_diff = False
                class_validation = None
                t1 = time.time()
            # else:
            #     current_gesture_record = 0
            #     t1 = time.time()
            #     class_validation = None
            #     frame_buffer = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
