import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
import numpy
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initilz. conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x

        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x

        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 1x1 -> 4x4
            nn.LeakyReLU(.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1,
                                    stride=1, padding=0)  # padding is 0 default, which is what we want
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList(
            [self.initial_rgb])  # we add the intial rgb bc we have that initial layer we dealt above

        for i in range(len(factors) - 1):
            # factors[i] -> factors[i+1], +! to ensure we dont index out the list, then we do -1
            conv_in_c = int(in_channels * factors[i])  # 1,1,1,1,1/2 , .. , 1/32
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    # steps ==0, (4x4), steps == 1 (8x8), .. we double at each of steps. steps ie current resolution we're on
    def forward(self, x, alpha, steps):
        out = self.initial(x)  # 4x4 right now

        if (steps == 0):
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        # making sure the rbg layer is working 1 behind, so it will be working on the same # of channels
        # run both of them thru 1x1 conv layer to make sure are both rbg channels
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(.2)

        # -1 : start at end of list, 0:move to 0, -1:steps of 1 going backwards
        # the first 1 we are appending is for the highest reso ex 1024, 512, ...
        # the last 1 will be the 8x8 because we already have that final block below for the 4x4 reso
        # * so, we just add the prog blocks in reverse order(start at top highest reso)
        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            # takes img channels to the in channels we have
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        # gonna have the 4x4 conv which isn't inside our conv block, similar to the initialize section for the generator
        # we wanna map rbg to 512 channels. So basically this for the 4x4 resolution
        # initial rbg takes img channels to in channels (which is 512)
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # append initial rbg because prog_block can sometimes change the # of channels, so the progblock needs to have 1 additional
        self.rgb_layers.append(self.initial_rgb)
        # this is the down sampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # block for 4x4 resolution (ie final block) takes it to a single value in the end
        self.final_block = nn.Sequential(
            # +1 because of the 513
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(.2),
            # this should be the same as that linear layer
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )
    # downscaled: what is out from the avg pooling. out: what is out from the conv layer
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        # this gives the std for every example of x. # N x C x H x W . gives std across all these, and = ->  N
        # .mean() gives us a single scalar value.                           height      width
        # repeat(): the single scalar value (from mean()) to be the exact same except its just 1 channel
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        # concatenating along the channels which will give us 512 -> 513
        return torch.cat([x, batch_statistics], dim=1)

    # what we want to do here is index correctly
    def forward(self, x, alpha, steps): # steps == 0 (4x4), steps == 1 (8x8), ...
        # when we are doing len, we are getting the total length. for ex. when steps is 1, we want the last one in the prog_blocks list/ModuleList
        # we get the len and subtract it by 1 and get the last index.
        cur_step = len(self.prog_blocks) - steps
        # run it thru rbg. we r using the rbg layer we appended earlier in __init__
        out = self.leaky(self.rgb_layers[cur_step](x))

        # now we have one more rbg we do for the blocks
        # to ensure we don't index incorrectly in the prog_blocks
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # recap: we have steps ex. 0, and we want 4x4 and that will be the end of the self.rbg_layer
        # becuase the end is where we have the smallest img size and begg. is the largest img sizes. bc reverse order
        # for the current step, which is the len of prog_blocks if its 4x4 (len of entire thing)
        # if it is 4x4 steps will be 0 and we send it to the minibatch_std to concat it and then run it thru the final block

        # from the fading in layer diagram: one is gonna run thru 0.5x(avg pooling) then to rbg
        # the other one is gonna go from rbg to progression blocks then to 0.5x

        # we avg pool, use rbg layers doing cur step +1 bc the progressive block might sometimes change the channel
        # we solve this by having rbg blocks 1 index away from the current one we're at
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))

        # so we run it thru the rbg in line 149, then run it thru the prog_blocks(line below), and then we avg pool
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # now we can do the fade in
        out = self.fade_in(alpha, downscaled, out)

        # +1 because we've already done the current step (the 3 statements before this)
        for step in range(cur_step + 1, len(self.prog_blocks)): # remember as we go up from cur block we make the img size smaller
            # two 3x3, downsample. two 3x3, downsample. .... repeats until end
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        # in the end is where we get 4x4 reso and the final block (stmt below) takes it to 1x1
        return self.final_block(out).view(out.shape[0], -1)

if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4,8,16,32,64,256,512,1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.rand((1, Z_DIM, 1, 1))
        z = gen(x, .5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=.5, steps=num_steps)
        assert out.shape == (1,1)
        print(f"success! at img size: {img_size}")
