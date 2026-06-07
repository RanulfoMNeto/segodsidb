"""
@brief  Module of evaluation measures or metrics.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com). 
@date   2 Jun 2021.
"""

import torch
import numpy as np
import monai.metrics
import torchvision
import scipy.optimize

# My imports
import torchseg.data_loader as dl


ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES = [
    "Skin", "Oral mucosa", "Enamel", "Tongue", "Lip", "Hard palate",
    "Attached gingiva", "Soft palate", "Hair", "Prosthetics",
]

ODSI_DB_UNMIXING_DEFAULT_RELEVANT_CLASS_NAMES = \
    ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()


def set_odsi_db_unmixing_relevant_class_names(class_names=None):
    """
    @brief Configure which ODSI-DB classes are used as semantic reference
           spectra for unmixing evaluation/mapping.
    @details This list is intentionally independent from R. For example,
             R can be 10 while only Enamel and Prosthetics are used as
             reference classes; the remaining estimated endmembers stay
             unmapped unless assigned to one of those references.
    """
    if class_names is None:
        class_names = ODSI_DB_UNMIXING_DEFAULT_RELEVANT_CLASS_NAMES

    class_names = [str(name) for name in class_names]
    if len(class_names) == 0:
        raise ValueError('At least one ODSI-DB unmixing reference class is '
                         'required.')
    if len(set(class_names)) != len(class_names):
        raise ValueError('Duplicate ODSI-DB unmixing reference classes: '
                         '{}'.format(class_names))

    valid_names = set(dl.OdsiDbDataLoader.OdsiDbDataset.classnames.values())
    invalid_names = [name for name in class_names if name not in valid_names]
    if invalid_names:
        raise ValueError('Unknown ODSI-DB unmixing reference classes: '
                         '{}'.format(invalid_names))

    ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES[:] = class_names
    return ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()


def configure_odsi_db_unmixing_from_config(config):
    """
    @brief Apply unmixing.reference_class_names from a config dictionary.
    @returns The active reference class names.
    """
    if hasattr(config, 'config'):
        config = config.config

    class_names = None
    if isinstance(config, dict):
        unmixing_config = config.get('unmixing', {})
        class_names = unmixing_config.get('reference_class_names')
        if class_names is None:
            class_names = unmixing_config.get('relevant_class_names')

    return set_odsi_db_unmixing_relevant_class_names(class_names)


def accuracy(pred, gt):
    """
    @brief Given a set of data points from repeated measurements of the same 
           quantity, the set can be said to be accurate if their average is 
           close to the true value of the quantity being measured.
    @param[in]  pred  TODO.
    @param[in]  gt    TODO.
    @returns TODO
    """
    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        assert pred.shape[0] == len(gt)
        correct = 0
        correct += torch.sum(pred == gt).item()
    return correct / len(gt)


def top_k_acc(pred, gt, k=3):
    """
    @brief Top-k accuracy means that the correct class gets to be in the top-k 
           predicted probabilities for it to count as correct.
    @param[in]  pred  TODO.
    @param[in]  gt    TODO.
    @returns TODO
    """
    with torch.no_grad():
        pred = torch.topk(pred, k, dim=1)[1]
        assert pred.shape[0] == len(gt)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == gt).item()
    return correct / len(gt)


def _reconstruction_from_output(pred):
    if isinstance(pred, dict):
        return pred['reconstruction']
    return pred


def sad_reconstruction_metric(pred, gt, eps=1e-8):
    with torch.no_grad():
        pred = _reconstruction_from_output(pred)
        pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1, eps=eps)
        gt_norm = torch.nn.functional.normalize(gt, p=2, dim=1, eps=eps)
        cosine = torch.sum(pred_norm * gt_norm, dim=1)
        cosine = torch.clamp(cosine, min=-1.0 + eps, max=1.0)
        cosine = torch.where(cosine > 1.0 - 1e-6,
                             torch.ones_like(cosine), cosine)
        return torch.acos(cosine).mean().item()


def rmse_reconstruction(pred, gt):
    with torch.no_grad():
        pred = _reconstruction_from_output(pred)
        return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


def reconstruction_min(pred, gt=None):
    with torch.no_grad():
        pred = _reconstruction_from_output(pred)
        return pred.min().item()


def reconstruction_max(pred, gt=None):
    with torch.no_grad():
        pred = _reconstruction_from_output(pred)
        return pred.max().item()


def reconstruction_mean(pred, gt=None):
    with torch.no_grad():
        pred = _reconstruction_from_output(pred)
        return pred.mean().item()


def target_reflectance_min(pred, gt):
    with torch.no_grad():
        return gt.min().item()


def target_reflectance_max(pred, gt):
    with torch.no_grad():
        return gt.max().item()


def target_reflectance_mean(pred, gt):
    with torch.no_grad():
        return gt.mean().item()


def asc_error(pred, gt=None):
    with torch.no_grad():
        abundances = pred['abundances']
        return torch.mean(torch.abs(torch.sum(abundances, dim=1) - 1.0)).item()


def abundance_nonnegativity_violation(pred, gt=None, tol=0.0):
    with torch.no_grad():
        abundances = pred['abundances']
        return torch.mean((abundances < -tol).float()).item()


def decoder_weight_nonnegativity_violation(pred, gt=None, tol=0.0):
    with torch.no_grad():
        endmembers = pred['endmembers']
        return torch.mean((endmembers < -tol).float()).item()


def endmember_min(pred, gt=None):
    with torch.no_grad():
        return pred['endmembers'].min().item()


def endmember_max(pred, gt=None):
    with torch.no_grad():
        return pred['endmembers'].max().item()


def endmember_mean(pred, gt=None):
    with torch.no_grad():
        return pred['endmembers'].mean().item()


def _odsi_db_relevant_class_indices(device):
    idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames
    class2idx = {y: x for x, y in idx2class.items()}
    return torch.tensor(
        [class2idx[x] for x in ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES],
        device=device, dtype=torch.long)


def odsi_db_unmixing_label_subset(labels, n_endmembers):
    """
    @brief Return labels used for semantic unmixing evaluation and the
           corresponding original ODSI-DB class indices.
    @details The configured relevant-class list is independent from R. This
             allows experiments such as R=10 with only Enamel/Prosthetics as
             reference classes. For non-ODSI label tensors, all channels are
             kept.
    """
    if labels.shape[1] == len(dl.OdsiDbDataLoader.OdsiDbDataset.classnames):
        class_indices = _odsi_db_relevant_class_indices(labels.device)
        return labels[:, class_indices, :, :], class_indices

    class_indices = torch.arange(labels.shape[1], device=labels.device,
                                 dtype=torch.long)
    return labels, class_indices


def odsi_db_unmixing_reference_sums(target_reflectance, labels,
                                    n_endmembers):
    """
    @brief Accumulate class reference spectra from annotated pixels.
    @details The returned spectra are not external library spectra; they are
             class-mean reflectance spectra computed from the labelled pixels
             available in the supplied split.
    @returns tuple (spectral_sums, pixel_counts, class_indices), where
             spectral_sums has shape (C_eval, bands).
    """
    labels, class_indices = odsi_db_unmixing_label_subset(
        labels, n_endmembers)

    valid = torch.sum(labels, dim=1, keepdim=True) == 1
    labels = labels.float() * valid.float()

    counts = labels.sum(dim=(0, 2, 3))
    spectral_sums = torch.einsum(
        'blhw,bkhw->kl', target_reflectance.float(), labels)
    return spectral_sums, counts, class_indices


def odsi_db_unmixing_reference_endmembers(target_reflectance, labels,
                                          n_endmembers):
    """
    @brief Compute class-mean reference spectra from labelled pixels.
    @returns tuple (reference_endmembers, pixel_counts, class_indices), where
             reference_endmembers has shape (C_eval, bands).
    """
    spectral_sums, counts, class_indices = \
        odsi_db_unmixing_reference_sums(target_reflectance, labels,
                                        n_endmembers)
    reference_endmembers = spectral_sums / counts.clamp_min(1.0).unsqueeze(1)
    return reference_endmembers, counts, class_indices


def odsi_db_unmixing_sad_cost_matrix(estimated_endmembers,
                                     reference_endmembers, eps=1e-8):
    """
    @brief Pairwise SAD cost matrix between estimated and reference spectra.
    @returns Tensor of shape (R, C_ref), in radians.
    """
    if torch.is_tensor(estimated_endmembers):
        estimated = estimated_endmembers.detach().float()
    else:
        estimated = torch.as_tensor(estimated_endmembers, dtype=torch.float32)

    if torch.is_tensor(reference_endmembers):
        reference = reference_endmembers.detach().float()
    else:
        reference = torch.as_tensor(reference_endmembers, dtype=torch.float32)

    if estimated.device != reference.device:
        reference = reference.to(estimated.device)

    estimated = torch.nn.functional.normalize(estimated, p=2, dim=1, eps=eps)
    reference = torch.nn.functional.normalize(reference, p=2, dim=1, eps=eps)
    cosine = estimated @ reference.transpose(0, 1)
    cosine = torch.clamp(cosine, min=-1.0 + eps, max=1.0)
    cosine = torch.where(cosine > 1.0 - 1e-6,
                         torch.ones_like(cosine), cosine)
    return torch.acos(cosine)


def odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
        estimated_endmembers, reference_endmembers, class_indices,
        reference_valid=None, max_sad=None, device=None,
        return_cost_matrix=False):
    """
    @brief Match estimated endmembers to class reference spectra using SAD.
    @details This is the classical HU-style permutation correction: rows are
             decoder endmembers, columns are reference spectra, and Hungarian
             minimizes the total spectral angle.
    @returns Tensor shape (R,) with original ODSI-DB class indices, or -1.
    """
    if torch.is_tensor(estimated_endmembers):
        estimated = estimated_endmembers.detach().float().cpu()
    else:
        estimated = torch.as_tensor(
            estimated_endmembers, dtype=torch.float32)

    if torch.is_tensor(reference_endmembers):
        reference = reference_endmembers.detach().float().cpu()
    else:
        reference = torch.as_tensor(
            reference_endmembers, dtype=torch.float32)

    if torch.is_tensor(class_indices):
        class_indices = class_indices.detach().cpu().numpy()
    else:
        class_indices = np.asarray(class_indices, dtype=np.int64)

    n_endmembers = estimated.shape[0]
    mapping = np.full((n_endmembers,), -1, dtype=np.int64)

    if estimated.numel() == 0 or reference.numel() == 0:
        mapping = torch.as_tensor(mapping, device=device, dtype=torch.long)
        if return_cost_matrix:
            return mapping, np.empty((n_endmembers, 0), dtype=np.float64)
        return mapping

    cost_matrix = odsi_db_unmixing_sad_cost_matrix(
        estimated, reference).cpu().numpy().astype(np.float64)

    if reference_valid is None:
        reference_valid = np.ones((reference.shape[0],), dtype=bool)
    elif torch.is_tensor(reference_valid):
        reference_valid = reference_valid.detach().cpu().numpy().astype(bool)
    else:
        reference_valid = np.asarray(reference_valid, dtype=bool)

    finite_cols = np.isfinite(cost_matrix).all(axis=0)
    valid_cols = np.where(np.logical_and(reference_valid, finite_cols))[0]
    if len(valid_cols) == 0:
        mapping = torch.as_tensor(mapping, device=device, dtype=torch.long)
        if return_cost_matrix:
            return mapping, cost_matrix
        return mapping

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
        cost_matrix[:, valid_cols])
    for row, valid_col_idx in zip(row_ind, col_ind):
        col = valid_cols[valid_col_idx]
        cost = cost_matrix[row, col]
        if max_sad is not None and cost > max_sad:
            continue
        mapping[row] = class_indices[col]

    mapping = torch.as_tensor(mapping, device=device, dtype=torch.long)
    if return_cost_matrix:
        return mapping, cost_matrix
    return mapping


def odsi_db_unmixing_spectral_hungarian_mapping(
        estimated_endmembers, target_reflectance, labels, max_sad=None):
    """
    @brief Compute a spectral endmember->class mapping from labelled spectra.
    """
    n_endmembers = estimated_endmembers.shape[0]
    reference_endmembers, counts, class_indices = \
        odsi_db_unmixing_reference_endmembers(
            target_reflectance, labels, n_endmembers)
    return odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
        estimated_endmembers, reference_endmembers, class_indices,
        max_sad=max_sad, device=estimated_endmembers.device)


def _odsi_db_unmixing_balanced_accuracy_from_mapping(abundances, labels,
                                                     mapping):
    n_endmembers = abundances.shape[1]
    labels, class_indices = odsi_db_unmixing_label_subset(
        labels, n_endmembers)

    valid = torch.sum(labels, dim=1) == 1
    if valid.sum().item() == 0:
        return 0.0

    mapping = torch.as_tensor(mapping, device=abundances.device,
                              dtype=torch.long)
    y_pred_endmember = torch.argmax(abundances, dim=1)[valid]
    y_pred_class = mapping[y_pred_endmember]
    y_true_relative = torch.argmax(labels, dim=1)[valid]
    y_true_class = class_indices[y_true_relative]

    scores = []
    for class_idx in class_indices.tolist():
        class_idx = int(class_idx)
        true_pos_class = y_true_class == class_idx
        pred_pos_class = y_pred_class == class_idx

        tp = torch.logical_and(true_pos_class, pred_pos_class).sum().float()
        tn = torch.logical_and(~true_pos_class, ~pred_pos_class).sum().float()
        fp = torch.logical_and(~true_pos_class, pred_pos_class).sum().float()
        fn = torch.logical_and(true_pos_class, ~pred_pos_class).sum().float()

        sens_den = tp + fn
        spec_den = tn + fp
        if sens_den.item() == 0 or spec_den.item() == 0:
            continue
        scores.append((0.5 * (tp / sens_den + tn / spec_den)).item())

    if not scores:
        return 0.0
    return float(np.mean(scores))


def odsi_db_unmixing_hungarian_balanced_accuracy(pred, gt=None):
    """
    @brief Post-hoc semantic evaluation for unmixing abundance maps.
    @details If pred contains semantic_mapping, that frozen mapping is used.
             Otherwise, when target reflectance is provided, endmembers are
             matched to class reference spectra by minimizing SAD. No
             abundance-mask overlap matching is performed.
    """
    with torch.no_grad():
        abundances = pred['abundances']
        labels = pred.get('label')
        if pred.get('disable_semantic_mapping', False):
            return 0.0
        if labels is None:
            return 0.0

        mapping = pred.get('semantic_mapping')
        if mapping is not None:
            return _odsi_db_unmixing_balanced_accuracy_from_mapping(
                abundances, labels, mapping)

        if gt is not None and 'endmembers' in pred:
            mapping = odsi_db_unmixing_spectral_hungarian_mapping(
                pred['endmembers'], gt, labels)
            return _odsi_db_unmixing_balanced_accuracy_from_mapping(
                abundances, labels, mapping)

        return 0.0


def iou(pred, gt, k=1):
    """
    @brief Computes the soft intersection over union for binary segmentation.
    @param[in]  pred  TODO.
    @param[in]  gt    TODO.
    @param[in]  k     Index of the positive class (starting from zero).
    @returns TODO.
    """
    with torch.no_grad():
        # TODO
        raise NotImplemented()
    return 0


def mean_iou(pred, gt, eps=1e-6, thresh=0.5):
    """
    @brief Computes the soft mean (over classes) intersection over union for all
           the images of the batch. Then all the mIoU per image are averaged over 
           all the images in the batch.
    @param[in]  pred  Tensor of predicted log-probabilities, shape (B, C, H, W).
    @param[in]  gt    Tensor of ground truth, shape (B, C, H, W).
    @returns a floating point with the loss value.
    """
    with torch.no_grad():
        # Flatten predictions and labels
        bs = pred.shape[0]
        chan = pred.shape[1] 
        flat_pred = torch.reshape(torch.exp(pred), (bs, chan, -1))
        flat_gt = torch.reshape(gt, (bs, chan, -1))

        # Binarise prediction
        bin_pred = (flat_pred > thresh).float()
        argmax_pred = torch.argmax(flat_pred, dim=1)
        one_hot_argmax_pred = torch.zeros_like(bin_pred)
        for i in range(bs):
            one_hot_argmax_pred[i] = torch.nn.functional.one_hot(argmax_pred[i], 
                num_classes=chan).transpose(0, 1)
        bin_pred *= one_hot_argmax_pred
        
        # Intersection 
        inter = torch.sum(bin_pred * flat_gt, dim=2)

        # Union
        union = torch.sum(bin_pred, dim=2) + torch.sum(flat_gt, dim=2) - inter
        
        # Mean IoU over classes
        miou = torch.mean((inter + eps) / (union + eps), dim=1)

        # Mean over the images in the batch
        batch_miou = torch.mean(miou).item()

    return batch_miou


def odsi_db_mean_iou(pred, gt, eps=1e-6, thresh=0.5):
    """
    @brief Computes the soft mean (over classes) intersection over union for all
           the images of the batch. Then all the mIoU per image are averaged over 
           all the images in the batch.
    @param[in]  pred  Tensor of predicted log-probabilities, shape (B, C, H, W).
    @param[in]  gt    Tensor of ground truth, shape (B, C, H, W).
    @returns a floating point with the loss value.
    """
    with torch.no_grad():
        # Flatten predictions and labels
        bs = pred.shape[0]
        chan = pred.shape[1] 
        flat_pred = torch.reshape(torch.exp(pred), (bs, chan, -1))
        flat_gt = torch.reshape(gt, (bs, chan, -1))

        # Binarise prediction
        flat_bin_pred = (flat_pred > thresh).float()
        argmax_pred = torch.argmax(flat_pred, dim=1)
        one_hot_argmax_pred = torch.zeros_like(flat_bin_pred)
        for i in range(bs):
            one_hot_argmax_pred[i] = torch.nn.functional.one_hot(argmax_pred[i], 
                num_classes=chan).transpose(0, 1)
        flat_bin_pred *= one_hot_argmax_pred

        # NOTE: Every image in the batch can have a different number of annotated
        #       pixels
        miou = torch.empty((bs))
        for i in range(bs):
            # Filter predictions without annotation, i.e. when all the ground 
            # truth classes are zero
            ann_idx = torch.sum(flat_gt[i, ...], dim=0) == 1
            y_true = flat_gt[i, :, ann_idx]
            y_pred = flat_bin_pred[i, :, ann_idx]
        
            # Intersection 
            inter = torch.sum(y_pred * y_true, dim=1)

            # Union
            union = torch.sum(y_pred, dim=1) + torch.sum(y_true, dim=1) - inter
        
            # Mean IoU over classes
            miou[i] = torch.mean((inter + eps) / (union + eps), dim=0)

        # Mean over the images in the batch
        batch_miou = torch.mean(miou).item()

    return batch_miou


def odsi_db_monai_metric(pred, gt, metric_name, ignore_labels=[], thresh=0.5):
    """
    @brief Computes the balanced accuracy averaged over all classes.
    @details True negatives: only the labelled pixels of the batch are counted. 
    @param[in]  pred           Tensor of predicted log-probabilities, 
                               shape (B, C, H, W).
    @param[in]  gt             Tensor of ground truth, shape (B, C, H, W).
    @param[in]  ignore_labels  List of classes (class names) that should be 
                               ignored when averaging the metric over classes.
    @param[in]  thresh         Theshold used to binarise predictions.
    @returns a scalar with the metric for the whole batch.
    """
    retval = None

    with torch.no_grad():
        # Flatten predictions and labels, and convert log-probabilities into
        # probabilities
        bs = pred.shape[0]
        chan = pred.shape[1] 
        height = pred.shape[2]
        width = pred.shape[3]
        flat_pred = torch.reshape(torch.exp(pred), (bs, chan, -1))
        flat_gt = torch.reshape(gt, (bs, chan, -1))
        
        # Binarise prediction
        flat_bin_pred = (flat_pred > thresh).float()
        
        # Shapes of bin_pred and flat_gt now are (B, C, H*W)
        assert(flat_bin_pred.shape == (bs, chan, height * width))
        assert(flat_gt.shape == (bs, chan, height * width))

        # NOTE: Every image in the batch can have a different number of 
        #       annotated pixels
        conf_mat = torch.zeros((1, chan, 4), device=pred.device)
        for i in range(bs):
            # Get tensor of booleans of valid annotated pixels (i.e. they
            # have been assigned to only one class)
            valid_idx = torch.sum(flat_gt[i, ...], dim=0) == 1

            # TODO: Add a case for when valid_idx is empty, i.e. what
            #       happens if the image does not have annotated pixels?
            
            # Get prediction and ground truth tensors, the shapes of the
            # tensors are (1, nclasses, npixels)
            y_true = flat_gt[i, :, valid_idx][None, :, :] 
            y_pred = flat_bin_pred[i, :, valid_idx][None, :, :]

            # Accumulate the confusion matrix over the images in the batch
            conf_mat += monai.metrics.get_confusion_matrix(y_pred, y_true)

        if metric_name == 'confusion matrix': 
            retval = conf_mat
        else:
            # Compute the metric with MONAI
            met = monai.metrics.compute_confusion_matrix_metric(metric_name, 
                conf_mat)[0]

            # We might want to ignore some classes for whatever reason
            idx2class = dl.OdsiDbDataLoader.OdsiDbDataset.classnames
            class2idx = {y: x for x, y in idx2class.items()}
            if ignore_labels:
                ignore_labels = [class2idx[x] for x in ignore_labels]
                for idx in ignore_labels:
                    met[idx] = float('nan') 

            # Mean metric over classes (ignoring classes whose metric is nan)
            is_nan = torch.isnan(met)
            met[is_nan] = 0
            mean_over_classes = (met.sum() / (~is_nan).float().sum()).item()

            retval = mean_over_classes

    return retval
         

def odsi_db_accuracy(pred, gt, ignore_labels=[]):
    return odsi_db_monai_metric(pred, gt, 'accuracy', 
        ignore_labels=ignore_labels)


def odsi_db_balanced_accuracy(pred, gt, ignore_labels=[]):
    sens = odsi_db_monai_metric(pred, gt, 'sensitivity', 
                                ignore_labels=ignore_labels)
    spec = odsi_db_monai_metric(pred, gt, 'specificity',
                                ignore_labels=ignore_labels)
    return .5 * sens + .5 * spec
    #return odsi_db_monai_metric(pred, gt, 'balanced accuracy',
    #    ignore_labels=ignore_labels)


def odsi_db_sensitivity(pred, gt, ignore_labels=[]):
    return odsi_db_monai_metric(pred, gt, 'sensitivity',
        ignore_labels=ignore_labels)


def odsi_db_specificity(pred, gt, ignore_labels=[]):
    return odsi_db_monai_metric(pred, gt, 'specificity',
        ignore_labels=ignore_labels)


def odsi_db_precision(pred, gt, ignore_labels=[]):
    return odsi_db_monai_metric(pred, gt, 'precision',
        ignore_labels=ignore_labels)


def odsi_db_f1_score(pred, gt, ignore_labels=[]):
    return odsi_db_monai_metric(pred, gt, 'f1 score',
        ignore_labels=ignore_labels)


def odsi_db_conf_mat(pred, gt, ignore_labels=[]):
    return odsi_db_monai_metric(pred, gt, 'confusion matrix',
        ignore_labels=ignore_labels)


def odsi_db_ResNet_18_CAM_DS_accuracy(tuple_pred, raw_gt, ignore_labels=[],
                                      thresh=0.5, pred_index=5):
    """
    @brief Class-presence accuracy metric for images.

    @details TP, TN, FP, FN explanation:

        * True positive:  the network estimates that the image contains pixels
                          of class K, and the image contains them.

        * True negative:  the network estimates that the image does not contain
                          pixels of class K, and the image does not contain 
                          them.

        * False positive: the network estimates that the image contains pixels
                          of class K, and the image does not contain them.
        
        * False negative: the network estimates that the image does not contain
                          pixels of class K, but the image contains them.

        Given an image, we have a vector of predictions, and a ground truth
        vector. Each element of the vector represents a class, a 1 means that
        the class is present in the image (i.e. there is at least one pixel
        belonging to this class), and a 0 means that the class is not present 
        in the image. 

        Based on these two vectors (prediction and ground truth) we can build
        a confusion matrix, and then compute an accuracy. This accuracy tells
        you how good is the system in predicting the classes that are present
        (and not present) in the image.
    
    @param[in]  tuple_pred  Tuple of six elements, the first one is a tensor 
                            containing the predicted class presence 
                            log-probabilities, shape (R, B, K, 1, 1).

                            * B is the batch size.

                            * R will be equivalent to the number of resolution 
                              levels + 1 (the global prediction). 

                            * K is the number of classes. 

                            The rest of the elements of the tuple are the
                            pseudo-segmentation predictions at different 
                            resolutions.

    @param[in]  raw_gt      Tensor of ground truth probabilities,
                            shape (B, C, H, W).

    @param[in]  thresh      Threshold to convert soft probability predictions  
                            into binary predictions.

    @param[in]  pred_index  tuple_pred[0] contains a tensor of dimensions 
                            (R, B, K, 1, 1), where each element of R contains 
                            a class-presence prediction at a different 
                            resolution.

                            The last item of R contains the global 
                            prediction. Using this parameter you can choose 
                            to use the class-presence prediction at any specific
                            resolution level (or the global one).
                            
                            By default, pred_index=5 because
                            that is where the global prediction is
                            stored by the forward pass of the model.

    @returns the average accuracy over the images in the batch.
    """
    retval = None

    with torch.no_grad():
        # Get the classification predictions, we will not use the
        # pseudo-segmentations to calculate the this metric
        raw_pred = tuple_pred[0]

        # Get dimensions
        bs = raw_pred.shape[1]
        classes = raw_gt.shape[1]
        rl = raw_pred.shape[0]  # Resolution levels + 1 (the global loss)
        
        # Loop over the batch
        acc = torch.cuda.FloatTensor(bs)
        for i in range(bs):
            # Build ground truth vector of class-presence probabilities from 
            # the segmentation ground truth that ODSI-DB provides
            gt = torch.cuda.FloatTensor(classes).fill_(0)
            for j in range(classes):
                gt[j] = 1 if 1 in raw_gt[i, j] else 0

            # Get the class-presence prediction vector, the five is to get the
            # global prediction (computed as the sum of the side predictions)
            soft_pred = torch.exp(torch.flatten(raw_pred[pred_index, i]))
            pred = (soft_pred > thresh).float()

            # Compute confusion matrix
            tp = torch.sum(torch.logical_and(pred == 1, gt == 1))
            tn = torch.sum(torch.logical_and(pred == 0, gt == 0))
            fp = torch.sum(torch.logical_and(pred == 1, gt == 0))
            fn = torch.sum(torch.logical_and(pred == 0, gt == 1))
            
            # Compute "accuracy" of class-presence detection for this 
            # batch image
            acc[i] = (tp + tn) / (tp + tn + fp + fn)

        # Compute the mean accuracy over the images of the batch
        retval = acc.mean().item()

        return retval


def convert_tuple_pred_to_segmap(tuple_pred, raw_gt, pred_index): 
    """
    @brief Convert segmentation feature maps into log-probability maps.
    """
    # Get "segmentation" feature maps 
    segmap = tuple_pred[1 + pred_index]  
    bs, classes, small_h, small_w = segmap.shape
    _, _, big_h, big_w = raw_gt.shape

    # Resize the "segmentation" feature maps to the original image size
    segmap = torchvision.transforms.Resize((big_h, big_w), 
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(segmap)
    
    # Apply sigmoid to convert the "segmentation" feature maps into 
    # "probability" maps
    segmap = torch.nn.functional.logsigmoid(segmap)

    return segmap


def odsi_db_ResNet_18_CAM_DS_pw_accuracy(tuple_pred, raw_gt, ignore_labels=[],
                                         pred_index=0):
    """
    @brief Evaluate the pixel-wise accuracy of the class-presence 
           classification model. 

    @param[in]  tuple_pred  See the comments in 
                            odsi_db_ResNet_18_CAM_DS_accuracy(). 

    @param[in]  raw_gt      See the comments in 
                            odsi_db_ResNet_18_CAM_DS_accuracy(). 

    @param[in]  pred_index  Indicates the resolution level where to take the
                            segmentation from, starting from zero with the
                            one of largest resolution.
    
    @details  The model and the method to obtain the segmentation is as 
              explained in:
           
              Garcia-Peraza Herrera et al. 2020 "Intrapapillary capillary loop 
              classification in magnification endoscopy: open dataset and 
              baseline methodology".
    """
    with torch.no_grad():
        segmap = convert_tuple_pred_to_segmap(tuple_pred, raw_gt, pred_index)
        return odsi_db_accuracy(segmap, raw_gt, ignore_labels)


def odsi_db_ResNet_18_CAM_DS_pw_balanced_accuracy(tuple_pred, raw_gt, 
                                                  ignore_labels=[],
                                                  pred_index=0):
    """
    @brief Evaluate the pixel-wise balanced accuracy of the class-presence 
           classification model. 
    
    @details  The model and the method to obtain the segmentation is as 
              explained in:
           
              Garcia-Peraza Herrera et al. 2020 "Intrapapillary capillary loop 
              classification in magnification endoscopy: open dataset and 
              baseline methodology".
    """
    with torch.no_grad():
        segmap = convert_tuple_pred_to_segmap(tuple_pred, raw_gt, pred_index)
        return odsi_db_balanced_accuracy(segmap, raw_gt, ignore_labels)


if __name__ == '__main__':
    raise RuntimeError('The metric.py module is not is not a script.') 
