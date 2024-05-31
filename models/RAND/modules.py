import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RAND.utils import FFT_for_Period, avgpool, sample
from rich import print

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()


    def forward(self, x_input: torch.Tensor):
        """
        args:
            x_input: shape is (b, length, dim)
        """
        u_input = x_input.mean(dim=1, keepdim=True)
        sigma_input_2 = torch.mean(torch.pow((x_input - u_input), 2),
                                   dim=1,
                                   keepdim=True)
        sigma_input = torch.sqrt(sigma_input_2)
        x_norm = (x_input - u_input) / sigma_input
        return x_norm

    def test_module(self):
        print(f"test Normalization")
        x_input = torch.randn((32, 336, 8))
        print(f"input_dim: {x_input.shape}")
        x_norm = self(x_input)
        print(f"output_dim: {x_norm.shape}")
        print(f"testsuccessful!!\n")

class LocalMultiPeriodicityExtractor(nn.Module):
    def __init__(self, m):
        super(LocalMultiPeriodicityExtractor, self).__init__()
        self.m = m

    def forward(self, x_input):
        """
        args:
            x_input: shape is (b, length, dim)
        """

        self.x_DFT = torch.fft.fft(x_input, dim=1)
        a = abs(self.x_DFT)
        self.a, f = torch.topk(a, dim=1, k=self.m)
        self.f = f + 1
        self.p = torch.ceil(self.x_DFT.shape[1] / self.f).to(torch.int64)
        return self.p

    def get_x_DFT(self):
        return self.x_DFT
    
    def get_a(self):
        return self.a
    
    def get_f(self):
        return self.f
    
    def get_p(self):
        return self.p

    def test_module(self):
        print(f"test LocalMultiPeriodicityExtractor")
        x_input = torch.randn((32, 336, 8))
        p = self(x_input)
        print(f"input_dim:")
        print(f"1. p.shape = {self.get_p().shape}")
        print(f"2. f.shape = {self.get_f().shape}")
        print(f"3. x_DFT.shape = {self.get_x_DFT().shape}")
        print(f"4. a.shape = {self.get_a().shape}")
        
        print(f"test successful!!\n")


class Decomposition(nn.Module):
    def __init__(self):
        super(Decomposition, self).__init__()
        pass

    def forward(self, x_input, x_DFT, a, p):
        """
        args:
            x_input: (b, T, d)
            x_DFT: (b, T, d)
            a: (b, m, d)
            p: (b, m, d)
        """
        # f = f.to(torch.int64)
        # f0 = f - 1
        # L = x_DFT.shape[1]

        # x_DFT_f = torch.gather(x_DFT, dim=1, index=f0)
        # x_DFT_Lk = torch.gather(x_DFT, dim=1, index = L-f)

        # x_DFT_f = x_DFT

        # x_DFT_f_irfft = torch.fft.ifft(x_DFT_f, dim=1, norm="backward")
        # x_DFT_Lk_irfft = torch.fft.ifft(x_DFT_Lk, dim=1, norm="backward")
        # x_sea = x_DFT_f_irfft + x_DFT_Lk_irfft

        x_sea = torch.fft.ifft(x_DFT, dim=1, norm="backward").real
        x_r = x_input - x_sea
        w_kernel = F.softmax(a, dim=1)   # (b, m, d)
        
        avgpool_xr = avgpool(x_r, p)    # (b, m, T, d)
        b, m, T, d = avgpool_xr.shape
        x_trend = (w_kernel.view(b, m, 1, d) * avgpool_xr).sum(dim=1)  # (b, T, d)
        x_R = x_input - x_trend
        return x_R, x_trend
    
    def test_module(self):
        model = LocalMultiPeriodicityExtractor(100)
        x_input = torch.randn((32, 336, 8))
        p = model(x_input)
        x_DFT = model.get_x_DFT()
        a = model.get_a()

        print(f"test Decomposition")
        x_R, x_trend = self(x_input, x_DFT, a, p)
        print(f"output_dim： x_R: {x_R.shape}  |  x_trend: {x_trend.shape}")
        print(f"testsuccessful!!\n")


class GlobalMultiPeriodicityExtractor(nn.Module):
    def __init__(self, m: int = 100):
        super(GlobalMultiPeriodicityExtractor, self).__init__()
        self.m = m

    def forward(self, xs):
        """
        args:
            xs: shape = (Ns, T, d)
        """
        x_DFT = torch.fft.fft(xs, dim=1)
        a = abs(x_DFT)
        a, f = torch.topk(a, dim=1, k=self.m)

        a, f = torch.topk(a, dim=1, k=self.m)
        f = f + 1    # (Ns, m, d)
        Ns, T, d = xs.shape
        repetitions = torch.zeros((T // 2, d)).to(xs.device)
        for ns in range(Ns):
            for i in range(self.m):
                repetitions[f[ns, i, :]] += 1

        repetitions /= (Ns * self.m)   # (T/2, d)
        return repetitions
        

    def test_module(self):
        xs = torch.randn((1000, 336, 8))
        xs = sample(xs, Ns=400)
        print(f"test：GlobalMultiPeriodicityExtractor")
        out = self(xs)
        print(f"output_dim： {out.shape}")
        print(f"testsuccessful!!!\n")


class ResolutionAdaptiveSlicing(nn.Module):
    def __init__(self, sql_len: int=336, L_slice: int=48, H: int=48):
        super(ResolutionAdaptiveSlicing, self).__init__()
        self.L_slice = L_slice
        input_dim = (sql_len // L_slice) * 6
        self.input_dim = input_dim
        self.mlp_u = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, H)
        )
        self.mlp_sigma = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, H)
        )


    def forward(self, x_input, x_trend, x_R, repetitions):
        """
        args:
            x_input: (b, L, d)
            x_trend: (b, L, d)
            x_R:     (b, L, d)
        """
        inputs = self.get_input(x_input, x_trend, x_R, repetitions)   #  shape = (b, 6N, d)
        inputs = inputs.permute(0, 2, 1)
        u_hat = self.mlp_u(inputs)
        sigma_hat = self.mlp_sigma(inputs)

        u_hat = u_hat.permute(0, 2, 1)       # shape = (b, 6N, out_dim)
        sigma_hat = sigma_hat.permute(0, 2, 1)
        return u_hat, sigma_hat
    
    def get_input(self, x_input, x_trend, x_R, repetitions):
        """
        args:
            x_input: (b, L, d)
            x_trend: (b, L, d)
            x_R:     (b, L, d)
        """
        repetitions = repetitions.to(torch.float32)
        T = x_input.shape[1]
        d = x_input.shape[-1]
        _lf = torch.arange(1, T//2+1).to(repetitions.device).view(-1, 1, 1)
        # print(repetitions.shape, _lf.shape)
        p_bar_weighted = (F.softmax(repetitions, dim=0) * _lf).sum()

        L_slice = self.cal_Lslice(p_bar_weighted, T, None)
        
        u_I, sigma_I = self.cal_u_sigma(x_input, L_slice)
        u_t, sigma_t = self.cal_u_sigma(x_trend, L_slice)
        u_R, sigma_R = self.cal_u_sigma(x_R, L_slice)
        # print(u_I.shape, sigma_I.shape, u_t.shape, sigma_t.shape, u_R.shape, sigma_R.shape)
        inputs = torch.concat((u_I, u_t, u_R, sigma_I, sigma_t, sigma_R), dim=1)
        return inputs
    
    def cal_u_sigma(self, x, L_slice):
        x_split = x.split(L_slice, dim=-2)  # tuple ((b, L_slice, d) ...)   N = T / L_slice
        x = torch.stack(x_split)   # (T/L_slice, b, L_slice, d)
        mean = x.mean(dim=-2)      # (T/L_slice, b, d)
        sigma = torch.mean(
            torch.pow(x - mean.unsqueeze(2), 2),
            dim=2
        )    # (T/L_slice, b, d)
        mean_split = torch.unbind(mean, dim=0)
        u = torch.stack(mean_split, dim=1)   # (b, T/L_slice, d)
        
        sigma_split = torch.unbind(sigma, dim=0)
        sigma = torch.stack(sigma_split, dim=1)

        return u, sigma
    
    
    def cal_Lslice(self, p_bar_weighted, L, H):
        return self.L_slice
    


    def test_module(self):
        xs = torch.randn((1000, 336, 8))
        net = GlobalMultiPeriodicityExtractor()
        repetitions = net(xs)


        model = LocalMultiPeriodicityExtractor(100)
        x_input = torch.randn((32, 336, 8))
        p = model(x_input)
        x_DFT = model.get_x_DFT()
        a = model.get_a()

        model = Decomposition()
        x_R, x_trend = model(x_input, x_DFT, a, p)
        
        
        print(f"repetitions: {repetitions.shape}")
        print(f"test：ResolutionAdaptiveSlicing")
        u, sigma = self(x_input, x_trend, x_R, repetitions)
        print(f"output_dim： u: {u.shape}   siamg: {sigma.shape}")
        print(f"testsuccessful!!\n")


class Extention(nn.Module):
    def __init__(self):
        super(Extention, self).__init__()

    def forward(self, u, sigma, H):
        """
        args:
            u:     shape = (b, out_dim, d)
            sigma: shape = (b, out_dim, d)
            H: output_dim
        """
        b, out_dim, d = u.shape
        assert H % out_dim == 0
        repeat = H // out_dim
        u_repeated = torch.repeat_interleave(u, repeat, dim=1)
        sigma_repeated = torch.repeat_interleave(sigma, repeat, dim=1)
        return u_repeated, sigma_repeated
    
    def test_module(self):
        xs = torch.randn((1000, 336, 8))
        net = GlobalMultiPeriodicityExtractor()
        repetitions = net(xs)


        model = LocalMultiPeriodicityExtractor(100)
        x_input = torch.randn((32, 336, 8))
        p = model(x_input)
        x_DFT = model.get_x_DFT()
        a = model.get_a()

        model = Decomposition()
        x_R, x_trend = model(x_input, x_DFT, a, p)

        model = ResolutionAdaptiveSlicing()
        u, sigma = model(x_input, x_trend, x_R, repetitions)
        print(f"test： Extention")
        u_out, sigma_out = self(u, sigma, 96)
        print(f"output_dim： u_out: {u_out.shape}  sigma_out: {sigma_out.shape}")
        print(f"testsuccessful!! \n")

class BackBone(nn.Module):
    def __init__(self, sql_len, H):
        super(BackBone, self).__init__()
        self.fc = nn.Linear(sql_len, H)

    def forward(self, x_input):
        x_input = x_input.permute(0, 2, 1)
        out = self.fc(x_input)
        out = out.permute(0, 2, 1)
        return out

class RAND(nn.Module):
    def __init__(self,
                 seq_len: int=336,
                 L_slice: int=48,
                 H: int=96,
                 m: int=100):
        """
        args:
            sql_len: length of sigle
            L_slice: 
            H: output_dim
            m: 
        """
        super(RAND, self).__init__()
        assert seq_len % L_slice == 0
        assert H % L_slice == 0
        self.H = H
        self.normalization = Normalization()
        self.local_multi_periodicity_extractor = LocalMultiPeriodicityExtractor(m)
        self.decomposition = Decomposition()
        self.global_multi_periodicity_extractor = GlobalMultiPeriodicityExtractor(m)
        self.resolution_adaptive_slicing = ResolutionAdaptiveSlicing(seq_len, L_slice, H)
        self.extention = Extention()
        self.backbone = BackBone(seq_len, H)

    def forward(self, x_input, trainset=None):
        breakpoint()
        x_norm = self.normalization(x_input)
        x_output = self.backbone(x_norm)
        p = self.local_multi_periodicity_extractor(x_input)
        x_DFT = self.local_multi_periodicity_extractor.get_x_DFT()
        a = self.local_multi_periodicity_extractor.get_a()
        x_R, x_trend = self.decomposition(x_input, x_DFT, a, p)

        # xs = torch.randn((1000, 336, 8))
        repetitions = self.global_multi_periodicity_extractor(trainset)
        u_hat, sigma_hat = self.resolution_adaptive_slicing(x_input, x_trend, x_R, repetitions)
        u_out, sigma_out = self.extention(u_hat, sigma_hat, self.H)
        
        y_out = x_output * sigma_out + u_out
        return y_out
    
    def test_module(self):
        x_input = torch.randn((32, 336, 8))
        print(f"test： RAND")
        trainset = torch.randn((1000, 336, 8))
        y_out = self(x_input, trainset)
        print(f"output_dim： {y_out.shape}")
        print(f"testsuccessful!!")


    
if __name__ == "__main__":
    # module = Normalization()
    # module.test_module()

    # module = LocalMultiPeriodicityExtractor(100)
    # module.test_module()

    # module = Decomposition()
    # module.test_module()

    # module = GlobalMultiPeriodicityExtractor()
    # module.test_module()

    # module = ResolutionAdaptiveSlicing()
    # module.test_module()

    # module = Extention()
    # module.test_module()

    trainset = torch.randn((1000, 336, 8))
    module = RAND()
    module.test_module()