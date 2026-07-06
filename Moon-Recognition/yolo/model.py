import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

#---------------------------------------------------------------------------------------
        
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
#-----------------------------------------------------------
class DFL(nn.Module):
    """Distribution Focal Loss (DFL) module."""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
#--------------last part
class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  
        self.nl = len(ch)  
        self.reg_max = 16  
        self.no = nc + self.reg_max * 4  

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        # c2 -> big enough for box regression (DFL needs room)
        # c3 -> big enough for class prediction (don't shrink too early)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:
            return x
        
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = self.dfl(box)
        return torch.cat((dbox, cls.sigmoid()), 1)
#---------------------
class DetectionModel(nn.Module):
    """YOLOv8n architecture replication."""
    def __init__(self, nc=80):
        super().__init__()
        
        # Wrapped in nn.Sequential to perfectly match the requested output root
        self.model = nn.Sequential(
            Conv(3, 16, 3, 2),                  # 0
            Conv(16, 32, 3, 2),                 # 1
            C2f(32, 32, 1, True),               # 2
            Conv(32, 64, 3, 2),                 # 3
            C2f(64, 64, 2, True),               # 4
            Conv(64, 128, 3, 2),                # 5
            C2f(128, 128, 2, True),             # 6
            Conv(128, 256, 3, 2),               # 7
            C2f(256, 256, 1, True),             # 8
            SPPF(256, 256, 5),                  # 9
            nn.Upsample(scale_factor=2.0, mode='nearest'), # 10
            Concat(1),                          # 11
            C2f(384, 128, 1, False),            # 12
            nn.Upsample(scale_factor=2.0, mode='nearest'), # 13
            Concat(1),                          # 14
            C2f(192, 64, 1, False),             # 15
            Conv(64, 64, 3, 2),                 # 16
            Concat(1),                          # 17
            C2f(192, 128, 1, False),            # 18
            Conv(128, 128, 3, 2),               # 19
            Concat(1),                          # 20
            C2f(384, 256, 1, False),            # 21
            Detect(nc, ch=[64, 128, 256])       # 22
        )

        self.save = [4, 6, 9, 12, 15, 18, 21]

    def forward(self, x):
        y = [] 
        # nn.Sequential can still be iterated through in PyTorch
        for i, m in enumerate(self.model):
            if i == 11: 
                x = m([x, y[6]])
            elif i == 14: 
                x = m([x, y[4]])
            elif i == 17: 
                x = m([x, y[12]])
            elif i == 20: 
                x = m([x, y[9]])
            elif i == 22: 
                x = m([y[15], y[18], x])
            else:
                x = m(x)
            
            y.append(x if i in self.save else None)
            
        return x

if __name__ == "__main__":
    model = DetectionModel()
    print(model)