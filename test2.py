import torch

def calc_nwd_tensor(bboxes1, bboxes2, eps=1e-6, constant=20):

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    area1 = area1.unsqueeze(-1).to(torch.double)  # [num_gt, 1]
    area2 = area2.unsqueeze(0) .to(torch.double) # [1, num_anchors]

    temp1 = torch.sqrt(area1)
    temp2 = torch.sqrt(area2)
    constant = torch.sqrt((area1 + area2 + eps)*2)  # [num_gt, num_anchors]
    eps = torch.tensor([eps])

    center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    whs = center1[..., :2] - center2[..., :2]

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

    w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
    h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
    w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
    h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wassersteins = torch.sqrt(center_distance + wh_distance)
    normalized_wassersteins = torch.exp(-wassersteins/12.8)

    return normalized_wassersteins


if __name__ == "__main__":
    bbox1 = torch.tensor([1, 1, 7, 7]).unsqueeze(0)
    bbox2 = torch.tensor([2, 2, 8, 8]).unsqueeze(0)
    bbox3 = torch.tensor([5, 5, 11, 11]).unsqueeze(0)
    nwd_12 = calc_nwd_tensor(bbox1, bbox2)
    nwd_13 = calc_nwd_tensor(bbox1, bbox3)
    print(nwd_12)
    print(nwd_13)
    bbox4 = torch.tensor([1, 1, 37, 37]).unsqueeze(0)
    bbox5 = torch.tensor([2, 2, 38, 38]).unsqueeze(0)
    bbox6 = torch.tensor([5, 5, 41, 41]).unsqueeze(0)
    nwd_45 = calc_nwd_tensor(bbox4, bbox5)
    nwd_46 = calc_nwd_tensor(bbox4, bbox6)
    print(nwd_45)
    print(nwd_46)