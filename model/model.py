import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F

from torchsummary import summary

from .conv_batchnorm_relu import ConvBatchnormRelu

class Model(nn.Module):
    def __init__(self, stage):
        super(Model, self).__init__()
        self.stage = stage

        #####################
        ####### Encode ######
        #####################
        self.conv1_1 = ConvBatchnormRelu(3, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.conv1_2 = ConvBatchnormRelu(64, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.conv2_1 = ConvBatchnormRelu(64, 128, kernel_size=3, padding=1,bias=True)
        self.conv2_2 = ConvBatchnormRelu(128, 128, kernel_size=3, padding=1,bias=True)
        self.conv3_1 = ConvBatchnormRelu(128, 256, kernel_size=3, padding=1,bias=True)
        self.conv3_2 = ConvBatchnormRelu(256, 256, kernel_size=3, padding=1,bias=True)
        self.conv3_3 = ConvBatchnormRelu(256, 256, kernel_size=3, padding=1,bias=True)
        self.conv4_1 = ConvBatchnormRelu(256, 512, kernel_size=3, padding=1,bias=True)
        self.conv4_2 = ConvBatchnormRelu(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv4_3 = ConvBatchnormRelu(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv5_1 = ConvBatchnormRelu(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv5_2 = ConvBatchnormRelu(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv5_3 = ConvBatchnormRelu(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv6_1 = ConvBatchnormRelu(512, 512, kernel_size=3, padding=1,bias=True)
        
        #####################
        ####### Trimap ######
        #####################
        self.trimap_deconv6_1 = ConvBatchnormRelu(512, 512, kernel_size=1,bias=True)
        self.trimap_deconv5_1 = ConvBatchnormRelu(512, 512, kernel_size=5, padding=2,bias=True)
        self.trimap_deconv4_1 = ConvBatchnormRelu(512, 256, kernel_size=5, padding=2,bias=True)
        self.trimap_deconv3_1 = ConvBatchnormRelu(256, 128, kernel_size=5, padding=2,bias=True)
        self.trimap_deconv2_1 = ConvBatchnormRelu(128, 64, kernel_size=5, padding=2,bias=True)
        self.trimap_deconv1_1 = ConvBatchnormRelu(64, 64, kernel_size=5, padding=2,bias=True)
        self.trimap_deconv1 = ConvBatchnormRelu(64, 3, kernel_size=5, padding=2,bias=True)

        #####################
        ####### Alpha  ######
        #####################
        self.deconv6_1 = ConvBatchnormRelu(1024, 512, kernel_size=1,bias=True)
        self.deconv5_1 = ConvBatchnormRelu(1024, 512, kernel_size=5, padding=2,bias=True)
        self.deconv4_1 = ConvBatchnormRelu(1024, 256, kernel_size=5, padding=2,bias=True)
        self.deconv3_1 = ConvBatchnormRelu(512, 128, kernel_size=5, padding=2,bias=True)
        self.deconv2_1 = ConvBatchnormRelu(256, 64, kernel_size=5, padding=2,bias=True)
        self.deconv1_1 = ConvBatchnormRelu(128, 64, kernel_size=5, padding=2,bias=True)
        self.deconv1 = ConvBatchnormRelu(128, 1, kernel_size=5, padding=2,bias=True)

        #####################
        ####### Refine ######
        #####################
        self.refine_conv1 = ConvBatchnormRelu(7, 64, kernel_size=3, padding=1, bias=True)
        self.refine_conv2 = ConvBatchnormRelu(64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_conv3 = ConvBatchnormRelu(64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_pred = ConvBatchnormRelu(64, 1, kernel_size=3, padding=1, bias=True)
        
        if stage == 'train_alpha':
            self.freeze_trimap_path()
        
    def forward(self, x):
        # Stage 1
        # x11 = self.conv1_1(x)
        # x12 = self.conv1_2(x11)
        x12 = self.conv1_2(self.conv1_1(x))
        s1 = x12.size()
        x1p, id1 = F.max_pool2d(x12,kernel_size=(2,2), stride=(2,2),return_indices=True)
        # Stage 2
        # x21 = self.conv2_1(x1p)
        # x22 = self.conv2_2(x21)
        x22 = self.conv2_2(self.conv2_1(x1p))
        s2 = x22.size()
        x2p, id2 = F.max_pool2d(x22,kernel_size=(2,2), stride=(2,2),return_indices=True)
        # Stage 3
        # x31 = self.conv3_1(x2p)
        # x32 = self.conv3_2(x31)
        # x33 = self.conv3_3(x32)
        x33 = self.conv3_3(self.conv3_2(self.conv3_1(x2p)))
        s3 = x33.size()
        x3p, id3 = F.max_pool2d(x33,kernel_size=(2,2), stride=(2,2),return_indices=True)
        # Stage 4
        # x41 = self.conv4_1(x3p)
        # x42 = self.conv4_2(x41)
        # x43 = self.conv4_3(x42)
        x43 = self.conv4_3(self.conv4_2(self.conv4_1(x3p)))
        s4 = x43.size()
        x4p, id4 = F.max_pool2d(x43,kernel_size=(2,2), stride=(2,2),return_indices=True)
        # Stage 5
        # x51 = self.conv5_1(x4p)
        # x52 = self.conv5_2(x51)
        # x53 = self.conv5_3(x52)
        x53 = self.conv5_3(self.conv5_2(self.conv5_1(x4p)))
        s5 = x53.size()
        x5p, id5 = F.max_pool2d(x53,kernel_size=(2,2), stride=(2,2),return_indices=True)
        # Stage 6
        x61 = self.conv6_1(x5p)
    


        # trimap 
        x6t = self.trimap_deconv6_1(x61)
        # Stage 5d
        x5t = torch.add(F.max_unpool2d(x6t, id5, kernel_size=2, stride=2, output_size=s5), x53)
        # x5t = x5t + x53
        # x51t = self.trimap_deconv5_1(x5t)
        # Stage 4d
        x4t = torch.add(F.max_unpool2d(self.trimap_deconv5_1(x5t), id4, kernel_size=2, stride=2, output_size=s4), x43)
        # x4t = x4t + x43
        # x41t = self.trimap_deconv4_1(x4t)
        # Stage 3d
        x3t = torch.add(F.max_unpool2d(self.trimap_deconv4_1(x4t), id3, kernel_size=2, stride=2, output_size=s3), x33)
        # x3t = x3t + x33
        # x31t = self.trimap_deconv3_1(x3t)
        # Stage 2d
        x2t = torch.add(F.max_unpool2d(self.trimap_deconv3_1(x3t), id2, kernel_size=2, stride=2, output_size=s2), x22)
        # x2t = x2t + x22
        # x21t = self.trimap_deconv2_1(x2t)
        # Stage 1d
        x1t = torch.add(F.max_unpool2d(self.trimap_deconv2_1(x2t), id1, kernel_size=2, stride=2, output_size=s1), x12)
        # x1t = x1t + x12
        # x12t = self.trimap_deconv1_1(x1t)
        raw_trimap = self.trimap_deconv1(self.trimap_deconv1_1(x1t))

        if self.stage == 'train_alpha':
            # Stage 6d
            x6d = self.deconv6_1(torch.cat((x61, x6t), 1))
            # Stage 5d
            x5d = torch.cat((torch.add(F.max_unpool2d(x6d, id5, kernel_size=2, stride=2, output_size=s5), x53), x5t), 1)
            # x5d = x5d + x53
            # x51d = self.deconv5_1(x5d)
            # Stage 4d
            x4d = torch.cat((torch.add(F.max_unpool2d(self.deconv5_1(x5d), id4, kernel_size=2, stride=2, output_size=s4), x43), x4t), 1)
            # x4d = x4d + x43
            # x41d = self.deconv4_1(x4d)
            # Stage 3d
            x3d = torch.cat((torch.add(F.max_unpool2d(self.deconv4_1(x4d), id3, kernel_size=2, stride=2, output_size=s3), x33), x3t), 1)
            # x3d = x3d + x33
            # x31d = self.deconv3_1(x3d)
            # Stage 2d
            x2d = torch.cat((torch.add(F.max_unpool2d(self.deconv3_1(x3d), id2, kernel_size=2, stride=2, output_size=s2), x22), x2t), 1)
            # x2d = x2d + x22
            # x21d = self.deconv2_1(x2d)
            # Stage 1d
            x1d = torch.cat((torch.add(F.max_unpool2d(self.deconv2_1(x2d), id1, kernel_size=2, stride=2, output_size=s1), x12), x1t), 1)
            # x1d = x1d + x12
            # x12d = self.deconv1_1(x1d)
            # Should add sigmoid? github repo add so.
            raw_alpha = self.deconv1(torch.cat((self.deconv1_1(x1d), x1t), 1))
            pred_mattes = torch.sigmoid(raw_alpha)

            # Stage2 refine conv1
            refine0 = torch.cat((x, raw_trimap, pred_mattes),  1)
            # refine1 = self.refine_conv1(refine0)
            # refine2 = self.refine_conv2(self.refine_conv1(refine0))
            # refine3 = self.refine_conv3(self.refine_conv2(self.refine_conv1(refine0)))
            # Should add sigmoid?
            # sigmoid lead to refine result all converge to 0... 
            #pred_refine = torch.sigmoid(self.refine_pred(refine3))
            pred_refine = self.refine_pred(self.refine_conv3(self.refine_conv2(self.refine_conv1(refine0))))
            pred_alpha = torch.sigmoid(torch.add(raw_alpha, pred_refine))

            #print(pred_mattes.mean(), pred_alpha.mean(), pred_refine.sum())
            return raw_trimap, pred_alpha
        return raw_trimap, None

    def freeze_trimap_path(self):
        for name, p in self.named_parameters():
            print(name)
            if name.startswith('conv') or name.startswith('trimap'):
                p.requires_grad = False
    def migrate(self, model):
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
        model_state_dict = checkpoint['model_state_dict']
        for name, p in self.named_parameters():
            if p.data.shape == model_state_dict[name].shape: 
                p.data = model_state_dict[name]
              
if __name__ == "__main__":
    model = EncodeNetwork()
    summary(model, (3, 320, 320))