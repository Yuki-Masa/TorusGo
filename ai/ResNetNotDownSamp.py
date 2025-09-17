import torch
import torch.nn as nn
from torchsummary import summary
import math

class Resblock(nn.Module): # ResNetのBottleneck Blockを模倣
    expansion = 4

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Resblock, self).__init__()

        # 全ての次元のストライドを常に1に強制
        # (depth_stride, width_stride, height_stride)
        self.stride = (1, 1, 1)

        # 1x1 conv
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        # 3x3 conv (空間次元のダウンサンプリングはここで行うが、全て1に固定)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 1x1 conv (チャネルの拡大)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)

        self.identity_downsample = identity_downsample # for shortcut connection

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out) # 最終ReLU

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, input_shape):
        super(ResNet, self).__init__()
        self.in_channels = 64 # ResNetの最初の畳み込み層の出力チャネル

        # 最初の畳み込みとプーリング
        # 全ての次元のストライドを1に固定
        self.conv1 = nn.Conv3d(input_shape[0], 64, kernel_size=7, stride=(1, 1, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        # MaxPoolもストライドを1に固定 (必要に応じてMax_poolを削除することも検討)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 1, 1), padding=1)

        self.stages = nn.ModuleList()
        current_in_channels = 64

        # ResNetの各ステージを作成
        for i, num_blocks in enumerate(layers):
            current_channels = 64 * (2 ** i) # 各ステージの出力チャネルを計算

            # 各ステージの最初のブロックでのストライド（ダウンサンプリング）
            # ここでも全ての次元のストライドを1に固定
            current_stride_dwh = (1, 1, 1)

            self.stages.append(self._make_stage(block, current_in_channels, current_channels, num_blocks, stride=current_stride_dwh))
            current_in_channels = current_channels * block.expansion

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) # グローバル平均プーリング
        self.fc = nn.Linear(current_in_channels, num_classes) # 最終の全結合層

    def _make_stage(self, block, in_channels, out_channels, num_blocks, stride):
        identity_downsample = None
        # ショートカット接続でのダウンサンプリングが必要な場合も、ストライドを1に固定
        if stride != (1,1,1) or in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * block.expansion, kernel_size=1, stride=(1,1,1), bias=False), # ストライドを1に固定
                nn.BatchNorm3d(out_channels * block.expansion)
            )

        layers = []
        # 各ステージの最初のブロックにのみストライドを適用 (ここもストライドは1に固定される)
        layers.append(block(in_channels, out_channels, identity_downsample, stride))

        # 残りのブロックはストライド1
        for _ in range(1, num_blocks):
            layers.append(block(out_channels * block.expansion, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        # このメソッドは、FC層を通る前の特徴マップを返す
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        # デフォルトのforwardパス (FC層まで含む)
        x = self.forward_features(x)
        #x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # Flatten
        #x = self.fc(x)
        return x

# ヘルパー関数: ResNetクラスを簡単にインスタンス化できるようにする
def get_CustomResNet(block, layers, num_classes, input_shape):
    return ResNet(block, layers, num_classes, input_shape)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 入力サイズを指定 (channels, depth, width, height)
    # widthとheightが小さい場合の例として (19, 19, 9, 9) を使用
    input_shape = (19, 19, 9, 9)
    num_classes = 1000 # ダミーのクラス数

    print(f"\nBuilding a ResNet with input shape: {input_shape}")

    # 例: 4ステージ (ResNet50に類似)
    layers_4_stages = [3, 4, 6, 3] # 各ステージのResBlock数
    print(f"Number of stages: {len(layers_4_stages)}")
    net_4_stages = get_CustomResNet(Resblock, layers_4_stages, num_classes, input_shape).to(device)

    # 空間次元の確認のためのダミー入力
    dummy_input = torch.randn(1, *input_shape).to(device)
    features_before_avgpool = net_4_stages.forward_features(dummy_input)
    print(f"Features before avgpool spatial shape: {features_before_avgpool.shape}")

    summary(net_4_stages, input_shape, device=str(device))
    print("---")
