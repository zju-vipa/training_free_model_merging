from torch.utils.data import Dataset, DataLoader, Subset
import os
import cv2
import torch
import numpy as np
import skimage
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import random
import itertools
from scipy.ndimage.filters import gaussian_filter
from visualpriors import visualpriors
from torch import nn
from tqdm import tqdm
import scipy
from visualpriors.visualpriors.transforms import VisualPriorPredictedLabel
from copy import deepcopy

taskonomy_data_dir = "./taskonomy_data"

def np_softmax(logits):
    maxs = np.amax(logits, axis=-1)
    softmax = np.exp(logits - np.expand_dims(maxs, axis=-1))
    sums = np.sum(softmax, axis=-1)
    softmax = softmax / np.expand_dims(sums, -1)
    return softmax


def resize_image(im, new_dims, interp_order=1):
    '''
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        interps = [PIL.Image.NEAREST, PIL.Image.BILINEAR]
        return skimage.util.img_as_float(im.resize(new_dims, interps[interp_order]))
    '''
    if all(new_dims[i] == im.shape[i] for i in range(len(new_dims))):
        resized_im = im  # return im.astype(np.float32)
    elif im.shape[-1] == 1 or im.shape[-1] == 3:
        resized_im = resize(im,
                            new_dims,
                            order=interp_order,
                            preserve_range=True)
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1, ), order=interp_order)
        # resized_im = resized_im.astype(np.float32)
    return resized_im


def random_noise_image(img, new_dims, new_scale, interp_order=1, seed=0):
    """
        Add noise to an image

        Args:
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
    Returns:
            a noisy version of the original clean image
    """
    img = skimage.util.img_as_float(img)
    img = resize_image(img, new_dims, interp_order)
    img = skimage.util.random_noise(img, var=0.01, seed=seed)
    img = rescale_image(img, new_scale)
    return img


def rescale_image(im, new_scale=[-1., 1.], current_scale=None, no_clip=False):
    im = skimage.img_as_float(im).astype(np.float32)
    # im = im.astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val)
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val
    return im


def resize_rescale_image(img,
                         new_dims,
                         new_scale=[-1, 1],
                         interp_order=1,
                         current_scale=None,
                         no_clip=False):
    img = skimage.img_as_float(img)
    img = resize_image(img, new_dims, interp_order)
    img = rescale_image(img,
                        new_scale,
                        current_scale=current_scale,
                        no_clip=no_clip)

    return img


def curvature_preprocess(img, new_dims, interp_order=1):
    img = resize_image(img, new_dims, interp_order)

    img = img[:, :, :2]
    # print("range curvature",np.max(img),np.min(img))
    img = img - [123.572, 120.1]
    img = img / [31.922, 21.658]
    return img


def resize_and_rescale_image_log(img, new_dims, offset=1., normalizer=1.):
    img = np.log(float(offset) + img) / normalizer
    img = resize_image(img, new_dims)
    return img


def load_raw_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Adapted from KChen

    Args:
        filename : string
        color : boolean
            flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).
    Returns
        image : an image with image original dtype and image pixel range
            of size (H x W x 3) in RGB or
            of size (H x W x 1) in grayscale.
    """

    try:
        img = skimage.io.imread(filename, as_gray=not color)
    except Exception as e:
        print("Wrong image path", filename)
        raise e

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def segment_pixel_sample(filename,
                         new_dims,
                         num_pixels,
                         domain,
                         mask=None,
                         is_aws=False):
    '''
     Segmentation
    Returns:
    --------
        pixels: size num_pixels x 3 numpy array
    '''
    img = skimage.io.imread(filename)
    # img = scipy.misc.imresize(img, tuple(new_dims), interp='nearest')

    if mask is None:
        all_pixels = list(
            itertools.product(range(img.shape[0]), range(img.shape[1])))
        pixs = random.sample(all_pixels, num_pixels)
    else:
        valid_pixels = list(zip(*np.where(np.squeeze(mask[:, :, 0]) != 0)))
        pixs = random.sample(valid_pixels, num_pixels)

    pix_segment = [list(i) + [int(img[i[0]][i[1]])] for i in pixs]
    pix_segment = np.array(pix_segment)
    return pix_segment


def semantic_segment_rebalanced(filename, new_dims, domain):
    '''
        Segmentation

        Returns:
        --------
            pixels: size num_pixels x 3 numpy array
        '''
    '''
    if template.split('/')[-1].isdigit():
        template = template.split('/')
        if template[0] == '':
            template[0] = os.sep
        template[-1] = "point_{point_id}_view_{view_id}_domain_{{domain}}.png".format(
            point_id=template[-2], view_id=template[-1])
        template[-2] = '{domain}'
        template = os.path.join(*template)
    filename = template.format(domain=domain)
    '''
    if not os.path.isfile(filename):
        print("no label")
        return np.zeros(tuple(new_dims)), np.zeros(tuple(new_dims))
    if os.stat(filename).st_size < 100:
        print("weong size")
        return np.zeros(tuple(new_dims)), np.zeros(tuple(new_dims))
    img = skimage.io.imread(filename)
    # img = scipy.misc.imresize(img, tuple(new_dims), interp='nearest')
    mask = img > 0.1
    mask = mask.astype(float)
    img[img == 0] = 1
    img = img - 1
    prior_factor = np.load('semseg_prior_factor.npy')
    # prior_factor = np.load(
    #     os.path.join("segsem_factor", domain, 'semseg_prior_factor.npy'))
    # prior_factor[0] = 0
    rebalance = prior_factor[img]
    mask = mask * rebalance
    mask = mask * (new_dims[0] * new_dims[1]) / mask.sum()
    return img, mask
    # return img, 1


def resize_rescale_image_gaussian_blur(img,
                                       new_dims,
                                       new_scale,
                                       interp_order=1,
                                       blur_strength=4,
                                       current_scale=None,
                                       no_clip=False):
    """
        Resize an image array with interpolation, and rescale to be
          between
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
        """
    img = skimage.img_as_float(img)
    img = resize_image(img, new_dims, interp_order)
    img = rescale_image(img,
                        new_scale,
                        current_scale=current_scale,
                        no_clip=True)
    blurred = gaussian_filter(img, sigma=blur_strength)
    if not no_clip:
        min_val, max_val = new_scale
        np.clip(blurred, min_val, max_val, out=blurred)
    return blurred


class TaskonomyDataset(Dataset):

    def __init__(self,
                 tasks=[],
                 transform=False,
                 root=taskonomy_data_dir,
                 domains=["ihlen"],
                 use_mask=True):
        """
        domain ihlen/     mcdade/    muleshoe/  noxapater/ uvalda/ 
        """
        print(f"data dir {root}")
        print(f"Contain domain {domains}")
        print(f"Contain tasks {tasks}")
        super().__init__()
        self.root = root
        self.images = []
        self.labels = [[] for _ in range(len(tasks))]
        self.tasks = tasks
        self.transform = transform
        self.domains = []
        self.depth_mask_list = [
            'edge_occlusion', 'keypoints2d', 'keypoints3d', 'reshading',
            "normal", 'curvature', 'depth_zbuffer', 'depth_euclidean'
        ] if use_mask else []
        self.use_depth_mask = any(t in self.depth_mask_list for t in tasks)
        print(f"using depth mask {self.use_depth_mask}")
        self.masks = []
        self.depth_norm = np.log(2.**16)
        for domain in domains:
            image_paths, label_paths, domains_, masks_ = self._load_image_path_from_domain(
                self.root, domain, tasks)
            self.images.extend(image_paths)
            self.domains.extend(domains_)
            self.masks.extend(masks_)
            for i, labels_ in enumerate(label_paths):
                self.labels[i].extend(labels_)

        for t, l in zip(self.tasks, self.labels):
            assert len(l) == len(
                self.images
            ), f"Wrong task labels number({t}): number of labels should be equal to number of images"

    def _load_image_path_from_domain(self, root, domain, tasks):
        root = os.path.join(root, domain)
        image_root = os.path.join(root, "rgb")
        images = [p for p in os.listdir(image_root)]

        def sort_key(x: str):
            x = x.split("_")
            return int(x[1]) * 1000000 + int(x[3])

        images = sorted(images, key=sort_key)
        image_paths = [os.path.join(image_root, p) for p in images]
        assert len(images) > 0, f"not find any images in {image_root}."
        label_roots = []
        for task in tasks:
            label_roots.append(os.path.join(root, self.get_task_dir(task)))

        labels_list = [[p for p in os.listdir(label_root)]
                       for label_root in label_roots]
        for i in range(len(labels_list)):
            labels_list[i] = sorted(labels_list[i], key=sort_key)
        label_paths = [[os.path.join(label_root, p) for p in labels]
                       for label_root, labels in zip(label_roots, labels_list)]
        # filter bad image
        image_paths = self.filter_bad_sample(image_paths)
        label_paths = [self.filter_bad_sample(ps) for ps in label_paths]
        for label_root, labels in zip(label_roots, label_paths):
            assert len(image_paths) == len(
                labels
            ), f"number of files in {label_root} should be equal to {image_root}"
        # domain_tag
        domains_ = [domain for _ in range(len(label_paths))]
        # masks
        mask_paths = []
        if self.use_depth_mask:
            masks_root = os.path.join(root, "depth_zbuffer")
            masks = [p for p in os.listdir(masks_root)]
            masks = sorted(masks, key=sort_key)
            mask_paths = [os.path.join(masks_root, p) for p in masks]
            mask_paths = self.filter_bad_sample(mask_paths)
            assert len(mask_paths) == len(
                image_paths
            ), f"number of mask in {masks_root} should be equal to {image_root}"
        return image_paths, label_paths, domains_, mask_paths

    def filter_bad_sample(self, paths):
        bad_samples = [("muleshoe", "point_399_view_5_domain")]
        return [
            p for p in paths
            if all((d not in p or s not in p) for d, s in bad_samples)
        ]

    def get_task_dir(self, task):
        if task == 'curvature':
            return 'principal_curvature'
        elif task in ["vanishing_point", 'room_layout']:
            return "point_info"
        elif task in ['denoising', 'autoencoding']:
            return 'rgb'
        return task

    def preprocess(self, image, labels):
        img = resize_rescale_image(image,
                                   new_dims=(256, 256),
                                   interp_order=1,
                                   current_scale=None,
                                   no_clip=False)

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.float()

        return img, labels

    def load_label(self,
                   task_name,
                   label_path,
                   cur_index=0,
                   domain=None,
                   depth_mask=1):
        current_scale_dict = {
            'edge_texture': [0, 0.08],
            'edge_occlusion': [0.0, 0.00625],
            'keypoints2d': [0.0, 0.005]
        }
        current_scale = None
        if task_name in current_scale_dict:
            current_scale = current_scale_dict[task_name]
        no_clip = False
        if task_name in ['edge_occlusion']:
            no_clip = True
        flag_img = 0
        if label_path.endswith(".png") or label_path.endswith(".jpg"):
            flag_img = 1
        elif label_path.endswith(".json"):
            flag_img = 2
        if flag_img == 1:
            # label = skimage.io.imread(label_path)

            if task_name == 'curvature':
                label = load_raw_image(label_path, color=True)
                label = curvature_preprocess(label,
                                             new_dims=(256, 256),
                                             interp_order=1)
            elif task_name in [
                    'keypoints2d', 'keypoints3d', 'reshading', 'edge_texture'
            ]:
                label = load_raw_image(label_path, color=False)
                label = resize_rescale_image(label,
                                             new_dims=(256, 256),
                                             interp_order=1,
                                             current_scale=current_scale,
                                             no_clip=no_clip)
            elif task_name in ['depth_zbuffer', 'depth_euclidean']:
                label = load_raw_image(label_path, color=False)
                label = resize_and_rescale_image_log(
                    label,
                    new_dims=(256, 256),
                    offset=1.,
                    normalizer=self.depth_norm)
            elif task_name in ['segment_unsup25d', 'segment_unsup2d']:
                label = segment_pixel_sample(filename=label_path,
                                             new_dims=(256, 256),
                                             num_pixels=300,
                                             domain=domain,
                                             mask=None,
                                             is_aws=False)
                return torch.from_numpy(label), 1
            elif task_name in ['segment_semantic']:
                label, mask = semantic_segment_rebalanced(filename=label_path,
                                                          new_dims=(256, 256),
                                                          domain=domain)
                return torch.from_numpy(label).long(), torch.from_numpy(
                    mask).float()
                # return torch.from_numpy(label).long(), 1
            elif task_name in ['edge_occlusion']:
                label = load_raw_image(label_path, color=False)
                label = resize_rescale_image_gaussian_blur(
                    label,
                    new_dims=(256, 256),
                    new_scale=[-1, 1],
                    interp_order=1,
                    blur_strength=4,
                    current_scale=current_scale,
                    no_clip=True)
            elif task_name in ["normal", 'autoencoding']:
                label = load_raw_image(label_path, color=True)
                label = resize_rescale_image(label,
                                             new_dims=(256, 256),
                                             interp_order=1,
                                             current_scale=current_scale,
                                             no_clip=no_clip)
            elif task_name in ['denoising']:
                label = load_raw_image(label_path, color=True)
                label = random_noise_image(label,
                                           new_dims=(256, 256),
                                           new_scale=[-1, 1],
                                           interp_order=1,
                                           seed=cur_index)
            else:
                raise NotImplementedError

            label = torch.from_numpy(label)
            label = label.permute(2, 0, 1)
            label = label.to(torch.float32)
        elif flag_img == 2:
            raise NotImplementedError
        else:
            label = np.load(label_path)
            label = torch.from_numpy(label).float()
        if task_name in self.depth_mask_list:
            depth_mask = torch.broadcast_to(depth_mask, label.shape)
        else:
            depth_mask = 1
        return label, depth_mask

    def __getitem__(self, index):
        image_path = self.images[index]
        img = load_raw_image(image_path, color=True)
        depth_mask = 1
        if self.use_depth_mask:
            depth_mask = load_raw_image(self.masks[index], color=False)
            depth_mask = (depth_mask < 64500).astype(float)
            depth_mask = resize_image(depth_mask, (256, 256), interp_order=1)
            depth_mask[depth_mask < 0.99] = 0
            depth_mask = torch.from_numpy(depth_mask).permute(2, 0, 1).float()
        labels = [
            self.load_label(t, l[index], domain=d, depth_mask=depth_mask)
            for t, l, d in zip(self.tasks, self.labels, self.domains)
        ]
        return self.preprocess(img, labels)

    def __len__(self):
        return len(self.images)


def gather_nd(params, indices):

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    out = torch.take(params, idx)

    return out.view(out_shape)


class TripleMetricLoss(nn.modules.loss._Loss):

    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(TripleMetricLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, batch_size=1):
        '''Returns the metric loss for 'num_pixels' embedding vectors.

        Args:
            output_imgs: Tensor of images output by the decoder.
            desired_imgs: Tensor of target images to be output by the decoder.
            masks: Tensor of masks to be applied when computing sum of squares
                    loss.

        Returns:
            losses: list of tensors representing each loss component
        '''
        output_vectors = input
        idx_segments = target
        #with tf.variable_scope('losses'):
        last_axis = 2
        #fir, sec, seg_id = tf.unstack(idx_segments, axis=last_axis)
        fir, sec, seg_id = torch.unbind(idx_segments, axis=last_axis)

        #idxes = tf.stack([self.batch_index_slice, fir, sec], axis=last_axis)

        num_pixels = fir.shape[1]
        idxes_out = np.asarray([range(batch_size)] * num_pixels).T
        idxes_out = torch.tensor(idxes_out, dtype=torch.float32)
        #batch_index_slice = torch.stack([idxes_out])
        batch_index_slice = idxes_out
        idxes = torch.stack([batch_index_slice, fir, sec], axis=last_axis)
        #self.embed = tf.gather_nd(output_vectors, idxes)
        #embed = self.embed
        embed = gather_nd(output_vectors, idxes)
        #square = tf.reduce_sum(embed * embed, axis=-1)
        square = torch.sum(embed * embed, dim=-1)
        #square_t = tf.expand_dims(square, axis=-1)
        square_t = torch.unsqueeze(square, dim=-1)
        #square = tf.expand_dims(square, axis=1)
        square = torch.unsqueeze(square, dim=1)

        #pairwise_dist = square - 2 * tf.matmul(embed, tf.transpose(embed, perm=[0, 2, 1])) + square_t
        #pairwise_dist = square - 2 * torch.matmul(embed, embed.permute(0,2,1)) + square_t
        pairwise_dist = square - 2 * torch.matmul(embed, embed.permute(
            1, 0)) + square_t
        #pairwise_dist = tf.clip_by_value(pairwise_dist, 0, 80)
        pairwise_dist = torch.clamp(pairwise_dist, 0, 80)

        # pairwise_dist = 0 - pairwise_dist 这个不改写
        #pairwise_exp = tf.exp(pairwise_dist) + 1
        pairwise_exp = torch.exp(pairwise_dist) + 1
        #sigma = tf.divide(2, pairwise_exp)
        sigma = torch.div(2, pairwise_exp)
        #sigma = tf.clip_by_value(sigma, 1e-7, 1.0 - 1e-7)
        sigma = torch.clamp(sigma, 1e-7, 1.0 - 1e-7)
        #self.sigma = sigma
        #same = tf.log(sigma)
        same = torch.log(sigma)
        #diff = tf.log(1 - sigma)
        diff = torch.log(1 - sigma)

        #seg_id_i = tf.tile(tf.expand_dims(seg_id, -1), [1, 1, self.num_pixels])
        x = torch.unsqueeze(seg_id, -1)
        seg_id_i = x.repeat(1, 1, 300)

        #seg_id_j = tf.transpose(seg_id_i, perm=[0, 2, 1])
        seg_id_j = seg_id_i.permute(0, 2, 1)

        #seg_comp = tf.equal(seg_id_i, seg_id_j)
        seg_comp = torch.equal(seg_id_i, seg_id_j)
        #seg_same = tf.cast(seg_comp, self.input_type)
        seg_same = torch.tensor(seg_comp, dtype=torch.float32)
        #seg_same = torch.FloatTensor(seg_comp) #input_type = float32
        seg_diff = 1 - seg_same

        loss_matrix = seg_same * same + seg_diff * diff
        #reduced_loss = 0 - tf.reduce_mean(loss_matrix)  # / self.num_pixels
        reduced_loss = 0 - torch.mean(loss_matrix)

        #tf.add_to_collection(tf.GraphKeys.LOSSES, reduced_loss)  可不用
        return reduced_loss


class ErrorRate(nn.modules.loss._Loss):

    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'none') -> None:
        super(ErrorRate, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        assert self.reduction == "none", "reduction should be none."
        return (torch.argmax(input.data, dim=1) != target).float()



def evaluate_model(models=None,
                    tasks=[],
                    domains=[],
                    root=taskonomy_data_dir,
                    batch_size=64,
                    num_workers=8):
    ds = TaskonomyDataset(tasks=tasks, domains=domains, root=root)
    softmax_task = [
        'class_object',
        # 'segment_semantic',
    ]
    errorrate_task = ['segment_semantic']
    l1loss_task = [
        'autoencoding', 'denoising', 'keypoints2d', 'keypoints3d', 'reshading',
        'depth_zbuffer', 'depth_euclidean', 'normal', 'edge_occlusion',
        'edge_texture'
    ]
    l2loss_task = ['curvature', 'room_layout', 'vanishing_point']
    triple_metric_loss_task = ['segment_unsup25d', 'segment_unsup2d']

    if models is None:
        print("Using pretrained models")
        models = visualpriors.load_models(feature_tasks=tasks)
    loss_fns = []
    losses = [0 for _ in range(len(tasks))]
    for t in tasks:
        if t in softmax_task:
            loss_fns.append(nn.CrossEntropyLoss(reduction="none"))
        elif t in l1loss_task:
            loss_fns.append(nn.L1Loss(reduction="none"))
        elif t in l2loss_task:
            loss_fns.append(nn.MSELoss(reduction="none"))
        elif t in errorrate_task:
            loss_fns.append(ErrorRate(reduction="none"))
        else:
            raise NotImplementedError

    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)
    for m in models:
        m.eval().cuda()

    total_pixel = [0 for _ in range(len(tasks))]
    for imgs, labels in tqdm(dl, total=len(dl), desc='Evaluating Loss'):
        imgs = imgs.cuda()
        for i, (m, (l, w)) in enumerate(zip(models, labels)):
            l = l.cuda()
            im = imgs

            if tasks[i] == "denoising":
                im, l = l, im

            output = m(im)
            # print("task", tasks[i], "label type", l.dtype, "shape", l.shape,
            #       "out shape", output.shape)
            raw_loss = loss_fns[i](output, l)
            if len(w.shape) > 1:
                # print("weight shape", w.shape, raw_loss.shape, tasks[i])
                w = w.cuda()
                raw_loss *= w
            total_pixel[i] += w.sum().cpu().detach().item()
            if len(raw_loss.shape) > 1 and len(w.shape) == 1:
                # if len(raw_loss.shape) > 1:
                raw_loss = raw_loss.flatten(1).mean(1)
            # print("raw loss shape", raw_loss.shape)

            losses[i] += raw_loss.sum().detach().cpu().item()

    losses = {t: loss / tp for t, loss, tp in zip(tasks, losses, total_pixel)}
    for m in models:
        m.cpu()
    return losses



def generate_avg_estimator(tasks=[],
                           domains=[],
                           root=taskonomy_data_dir,
                           batch_size=64,
                           num_workers=8,
                           save_root="weights/avg_estimator"):
    print("Generating domains", domains)
    for d in domains:
        print("Generating domain", d)
        tasks_ = []
        for t in tasks:
            save_path = os.path.join(save_root, d, t)
            file_path = os.path.join(save_path, "avg.pth")
            tasks_.append(t)
            if not os.path.exists(file_path):
                tasks_.append(t)
            else:
                print(f"Existing domain {d} task {t} at", file_path)
        print("Generating tasks", tasks_)
        if len(tasks_) <= 0:
            print("Not finded task in domain", d)
            continue
        avg_estimator = {k: 0 for k in tasks_}
        ds = TaskonomyDataset(tasks=tasks_,
                              domains=[d],
                              root=root,
                              use_mask=False)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)
        for imgs, labels in tqdm(dl, total=len(dl)):
            for i, (l, w) in enumerate(labels):
                if tasks_[i] == "denoising":
                    l = imgs

                l.cuda()
                if tasks_[i] == 'segment_semantic':
                    l = torch.nn.functional.one_hot(l,
                                                    num_classes=17).transpose(
                                                        -1, 1)
                avg_estimator[tasks_[i]] = l.sum(0).detach()

        for t in tasks_:
            save_path = os.path.join(save_root, d, t)
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, "avg.pth")
            avg = (avg_estimator[t] / len(ds)).cpu()
            torch.save(avg, file_path)


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


def prepare_resetbns_dataloader(domains=[], batch_size=64, num_workers=8):
    ds = TaskonomyDataset(domains=domains,
                          root=taskonomy_data_dir,
                          use_mask=False)
    generator = torch.Generator()
    generator.manual_seed(0)
    indices = torch.randperm(len(ds), generator=generator)
    ds = Subset(ds, indices=indices)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)


def reset_bn_stats(models, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    if isinstance(models, nn.Module):
        models = [models]

    device = get_device(models[0])
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for model in models:
        for m in model.modules():
            if type(m) == nn.BatchNorm2d:
                if reset:
                    m.momentum = None  # use simple average
                    m.reset_running_stats()
                has_bn = True

    if not has_bn:
        return models

    # run a single train epoch with augmentations to recalc stats
    for model in models:
        model.train()
    with torch.no_grad():
        for images, _ in tqdm(
                loader, desc=f'Resetting batch norm number {len(models)}'):
            for model in models:
                _ = model(images.to(device))
    return models


def load_resetbns_models(feature_tasks=["normal"],
                         loader=None,
                         cache_root="weights/resetbns"):
    VisualPriorPredictedLabel._load_unloaded_nets(feature_tasks)
    nets = [
        VisualPriorPredictedLabel.feature_task_to_net[t] for t in feature_tasks
    ]
    nets = [deepcopy(m) for m in nets]
    unload_models = {}
    for i, t in enumerate(feature_tasks):
        task_path = os.path.join(cache_root, t)
        os.makedirs(task_path, exist_ok=True)
        state_dict_path = os.path.join(task_path, "ckpt.pth")
        if os.path.exists(state_dict_path):
            nets[i].load_state_dict(
                torch.load(state_dict_path, map_location="cpu"))
            print(f"Loaded resetbns model({t}) from {state_dict_path}")
        else:
            unload_models[t] = nets[i]
    if len(unload_models) > 0:
        for m in unload_models.values():
            m.cuda()
        reset_bn_stats([x for x in unload_models.values()], loader)
        for t, m in unload_models.items():
            task_path = os.path.join(cache_root, t)
            os.makedirs(task_path, exist_ok=True)
            state_dict_path = os.path.join(task_path, "ckpt.pth")
            torch.save(m.cpu().state_dict(), state_dict_path)
            print(f"Saved resetbns model({t}) to {state_dict_path}")
    return nets


def randomize_model(model: nn.Module):
    for n, w in model.state_dict().items():
        if n.endswith("weight") or n.endswith("bias"):
            nn.init.normal_(w, 0, 0.01)
        elif n.endswith("running_mean"):
            w.fill_(0)
        elif n.endswith("running_var"):
            w.fill_(1)


def load_random_models(feature_tasks=["normal"],
                       loader=None,
                       cache_root="randmodel"):
    VisualPriorPredictedLabel._load_unloaded_nets(feature_tasks)
    nets = [
        VisualPriorPredictedLabel.feature_task_to_net[t] for t in feature_tasks
    ]
    nets = [deepcopy(m) for m in nets]
    unload_models = {}
    for i, t in enumerate(feature_tasks):
        task_path = os.path.join(cache_root, t)
        os.makedirs(task_path, exist_ok=True)
        state_dict_path = os.path.join(task_path, "ckpt.pth")
        if os.path.exists(state_dict_path):
            nets[i].load_state_dict(
                torch.load(state_dict_path, map_location="cpu"))
            print(f"Loaded random model({t}) from {state_dict_path}")
        else:
            m = nets[i]
            randomize_model(m)
            unload_models[t] = m
    if len(unload_models) > 0:
        for m in unload_models.values():
            m.cuda()
        reset_bn_stats([x for x in unload_models.values()], loader)
        for t, m in unload_models.items():
            task_path = os.path.join(cache_root, t)
            os.makedirs(task_path, exist_ok=True)
            state_dict_path = os.path.join(task_path, "ckpt.pth")
            torch.save(m.cpu().state_dict(), state_dict_path)
            print(f"Saved random model({t}) to {state_dict_path}")
    return nets


class AvgEstimator(nn.Module):

    def __init__(self, domain, task, avg_cache="") -> None:
        super().__init__()
        path = os.path.join(avg_cache, domain, task, "avg.pth")
        self.avg_label: torch.Tensor = torch.load(path)
        self._xxx = [1 for _ in range(len(self.avg_label.shape))]

    def forward(self, x):
        batch_size = x.shape[0]
        return self.avg_label.unsqueeze(0).repeat(batch_size, *self._xxx).to(x)