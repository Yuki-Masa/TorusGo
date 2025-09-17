import torch
import torch.nn as nn
from torchsummary import summary
import math

class Resblock(nn.Module): # ResNetのBottleneck Blockを模倣
    expansion = 4 # Bottleneckでは出力チャネルが入力チャネルの4倍になる

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Resblock, self).__init__()

        # ストライドがintの場合、(stride, stride)のタプルに変換（2D用）
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride # 渡されたstrideを保持

        # 1x1 conv: 入力チャネルをout_channelsに削減 (ボトルネック)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 3x3 conv (空間次元のダウンサンプリングはここで行う)
        # stride をここで適用
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv (チャネルの拡大): out_channels * expansion に拡大
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.identity_downsample = identity_downsample # ショートカット接続
        # self.stride は上記で設定済み

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) # 論文のResNet v1.5に合わせる（conv3の前にReLU）

        x = self.conv3(x)
        x = self.bn3(x)

        # ショートカット接続
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x) # 最終的なReLU

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, input_shape, uniform_channel_width=256):
        super(ResNet, self).__init__()
        self.block = block
        self.expansion = block.expansion

        initial_channels, initial_height, initial_width = input_shape # 2D入力形状 (channels, height, width)
        self.current_height, self.current_width = initial_height, initial_width

        # --- Initial layers ---
        # conv1の出力チャネルもuniform_channel_widthに設定
        self.in_channels = uniform_channel_width
        self.conv1 = nn.Conv2d(initial_channels, self.in_channels, kernel_size=7, stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

        # conv1とmaxpool後の次元を更新 (2D用)
        self.current_height = math.floor((self.current_height + 2 * 3 - 7) / 2) + 1 # conv1 stride 2
        self.current_width = math.floor((self.current_width + 2 * 3 - 7) / 2) + 1

        self.current_height = math.floor((self.current_height + 2 * 1 - 3) / 2) + 1 # maxpool stride 2
        self.current_width = math.floor((self.current_width + 2 * 1 - 3) / 2) + 1

        print(f"After conv1 and maxpool: spatial_dims=({self.current_height}, {self.current_width}), channels={self.in_channels}")

        # --- Residual Stages ---
        self.stages = nn.ModuleList()
        # 各残差ブロック内部の畳み込み層のチャネル数（拡張前のチャネル数）
        block_internal_channels = uniform_channel_width // self.expansion

        for i, num_resblocks in enumerate(layers):
            stage_name = f"layer{i+1}"

            current_stride = (1, 1) # デフォルトはダウンサンプリングなし (2D用)

            # 動的なダウンサンプリングの決定ロジック (2D用)
            # 空間次元がまだ十分大きく（例: 5より大きい）、かつダウンサンプリングで次元が3以上を保てる場合
            next_h = max(1, math.floor((self.current_height + 2*1 - 3)/2) + 1) # stride=2, kernel=3, pad=1の場合
            next_w = max(1, math.floor((self.current_width + 2*1 - 3)/2) + 1)

            should_downsample_h = (next_h >= 3) and (self.current_height > 5)
            should_downsample_w = (next_w >= 3) and (self.current_width > 5)

            stride_h = 2 if should_downsample_h else 1
            stride_w = 2 if should_downsample_w else 1

            current_stride = (stride_h, stride_w)

            # 各ステージの最初のブロックでチャネル数がuniform_channel_widthになるように、
            # block_internal_channelsを渡す
            stage = self._make_layer(block, num_resblocks, num_filters_internal_block=block_internal_channels, stride=current_stride, name=stage_name)
            self.stages.append(stage)

            # ストライドが適用された場合、現在の空間次元を更新 (2D用)
            if current_stride != (1, 1):
                self.current_height = math.floor((self.current_height + 2*1 - 3) / current_stride[0]) + 1
                self.current_width = math.floor((self.current_width + 2*1 - 3) / current_stride[1]) + 1
                print(f"After {stage_name} downsample: spatial_dims=({self.current_height}, {self.current_width}), channels={self.in_channels}")
            else:
                 print(f"After {stage_name} (no downsample): spatial_dims=({self.current_height}, {self.current_width}), channels={self.in_channels}")

        # --- Final layers ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 2D用
        # self.in_channelsは最後のステージの出力チャネル数（uniform_channel_width）になっている
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, num_resblocks, num_filters_internal_block, stride, name):
        identity_downsample = None
        current_in_channels = self.in_channels # この層への入力チャネル数

        # ショートカット接続が必要な条件: 空間次元がダウンサンプリングされるか、チャネル数が変化する場合
        if stride != (1, 1) or current_in_channels != num_filters_internal_block * self.expansion: # 2D用
            identity_downsample = nn.Sequential(
                nn.Conv2d(current_in_channels, num_filters_internal_block * self.expansion, kernel_size=1, stride=stride, bias=False), # Conv2d
                nn.BatchNorm2d(num_filters_internal_block * self.expansion) # BatchNorm2d
            )

        layers_in_stage = []
        # ステージの最初のブロック（ストライドが適用される可能性がある）
        layers_in_stage.append(block(current_in_channels, num_filters_internal_block, identity_downsample, stride))

        # 最初のブロックの出力チャネル数が、その後のブロックの入力チャネル数になる
        self.in_channels = num_filters_internal_block * self.expansion

        # ステージ内の残りのブロック（ストライド1で、チャネル数も変化しない）
        for i in range(1, num_resblocks):
            layers_in_stage.append(block(self.in_channels, num_filters_internal_block))

        return nn.Sequential(*layers_in_stage)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        # 特徴抽出部分
        x = self.forward_features(x)

        # 平均プーリング
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # Flatten

        # 全結合層 (分類ヘッド)
        x = self.fc(x)
        return x

# ヘルパー関数: ResNetクラスを簡単にインスタンス化できるようにする
def get_CustomResNet(block, layers, num_classes, input_shape, uniform_channel_width=256):
    return ResNet(block, layers, num_classes, input_shape, uniform_channel_width)

"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 入力サイズを指定 (チャンネル数, 高さ, 幅) - 2D用
    input_shapes_to_test = [(16, 9, 9)] # 奥行きを削除
    num_classes = 1000 # ダミーのクラス数

    for input_shape in input_shapes_to_test:
        print(f"\n--- Testing with input shape: {input_shape} ---")

        # 例1: 4ステージ (ResNet50に類似) - uniform_channel_width=256
        layers_4_stages = [3, 4, 6, 3] # 各ステージのResBlock数
        print(f"Number of stages: {len(layers_4_stages)}, uniform_channel_width=256")
        net_4_stages = get_CustomResNet(Resblock, layers_4_stages, num_classes, input_shape, uniform_channel_width=256).to(device)
        summary(net_4_stages, input_shape, device=str(device))
        print("---")

        # 例2: 9ステージ - uniform_channel_width=384 (KataGoのハイエンドモデルに類似)
        layers_9_stages = [2, 2, 2, 2, 2, 2, 2, 2, 2] # 各ステージのResBlock数 (9ステージ)
        print(f"Number of stages: {len(layers_9_stages)}, uniform_channel_width=384")
        net_9_stages = get_CustomResNet(Resblock, layers_9_stages, num_classes, input_shape, uniform_channel_width=384).to(device)
        summary(net_9_stages, input_shape, device=str(device))
        print("---")
"""
