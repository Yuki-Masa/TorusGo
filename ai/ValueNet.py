import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary # torchsummary がない場合は pip install torchsummary でインストール

# ResNetDynamicAlpha.py は同じディレクトリにあると仮定します。
from ResNetDynamicAlpha import get_CustomResNet, Resblock

class Custom3DNetScalarOutput(nn.Module):
    def __init__(self,
                 input_shape: tuple,  # (channels, depth, width, height)
                 output_spatial_shape: tuple, # (depth, width, height) - Adaptive Poolingの出力空間形状
                 resnet_layers: list, # ResNetの各ステージのブロック数を定義するリスト (例: [3, 4, 6, 3])
                 num_classes_for_resnet: int):
        """
        カスタム3Dニューラルネットワーク。
        ResNet -> Conv3d -> BatchNorm3d -> ReLU -> AdaptiveAvgPool3d -> Flatten -> Dense -> Dense -> Tanh の構成。
        最終的な出力は -1から1 のスカラー値となる。

        Args:
            input_shape (tuple): 入力データの形状 (channels, depth, width, height)。
            output_spatial_shape (tuple): AdaptiveAvgPool3d の目的の出力空間形状 (depth, width, height)。
            resnet_layers (list): get_CustomResNet関数に渡す、各ステージのブロック数を定義するリスト。
            num_classes_for_resnet (int): ResNetモデルの初期化に必要ですが、
                                         このネットワークではResNetの最終FC層は使用しません。
        """
        super(Custom3DNetScalarOutput, self).__init__()

        # 入力形状の展開
        in_channels, in_depth, in_width, in_height = input_shape
        # Adaptive Poolingの出力空間形状の展開
        self.out_depth, self.out_width, self.out_height = output_spatial_shape

        # 最終出力は-1~1のスカラーなので次元は1
        self.output_dimension = 1

        # 添付のget_CustomResNet関数を使ってResNetモデルをインスタンス化
        self.resnet = get_CustomResNet(
            block=Resblock,
            layers=resnet_layers,
            num_classes=num_classes_for_resnet,
            input_shape=input_shape
        )

        # ResNetの出力チャネル数を特定するためのダミー推論
        dummy_resnet_input = torch.randn(1, *input_shape)
        with torch.no_grad(): # 勾配計算を無効化
            dummy_resnet_output = self.resnet.forward_features(dummy_resnet_input)

        resnet_output_channels = dummy_resnet_output.shape[1]

        # 中間層のConv3d
        self.conv3d = nn.Conv3d(in_channels=resnet_output_channels,
                                out_channels=64, # 任意の中間チャネル数
                                kernel_size=(3, 3, 3),
                                padding=(1, 1, 1))

        self.bn3d = nn.BatchNorm3d(64)
        self.relu = nn.ReLU() # conv3dの後に使用

        # Adaptive poolingで出力の空間次元をoutput_spatial_shapeに調整
        self.adaptive_pool = nn.AdaptiveAvgPool3d((self.out_depth, self.out_width, self.out_height))

        # Flatten後のLinear層の入力特徴量を計算
        # adaptive_poolの出力形状は (B, 64, self.out_depth, self.out_width, self.out_height)
        flattened_features = 64 * self.out_depth * self.out_width * self.out_height

        # ユーザーの要求に基づく新しい層
        # (flatten+dense) -> (dense+tanh)
        self.fc1 = nn.Linear(flattened_features, 512) # 中間層の次元は任意
        self.fc2 = nn.Linear(512, self.output_dimension)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ネットワークの順伝播。

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, channels, depth, width, height)。

        Returns:
            torch.Tensor: 出力テンソル (batch_size, 1) の形式で、
                          -1から1の間の連続値（スカラー）。
        """
        # ResNetによる特徴抽出
        resnet_features = self.resnet.forward_features(x)

        # Conv3d -> BatchNorm3d -> ReLU
        x = self.conv3d(resnet_features)
        x = self.bn3d(x)
        x = self.relu(x) # Conv3dの後にReLU適用

        # Adaptive poolingで空間次元を目的の出力サイズに合わせる
        x = self.adaptive_pool(x) # (B, 64, D, W, H)

        # Flatten
        x = x.view(x.size(0), -1) # (B, 64 * D * W * H)

        # Dense (Linear) 層
        x = self.fc1(x)

        # Dense (Linear) 層 + Tanh
        x = self.fc2(x)
        x = self.tanh(x) # (B, 1)

        # 最終出力はスカラーなので、形状は (batch_size, 1) のまま

        return x

# ResNetクラスの修正点 (ResNetDynamicAlpha.pyにこれを追加してください)
"""
class ResNet(nn.Module):
    # ... 既存のコード ...

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
        return x # flattenされる前の特徴マップを返す

    def forward(self, x):
        # 既存のforwardメソッドはそのままにしておくか、
        # forward_featuresを呼び出してFC層に渡すように変更することも可能
        x = self.forward_features(x)
        # x = self.avgpool(x) # もしavgpoolとfc層を使用するなら
        x = x.reshape(x.shape[0], -1) # flatten
        # x = self.fc(x) # もしfc層を使用するなら
        return x
"""

# --- 訓練コード ---

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train() # モデルを訓練モードに設定
    for epoch in range(num_epochs):
        running_loss = 0.0
        target = train_loader[0]
        target = target.to(device)
        del train_loader[0]
        for input, _, player in train_loader:
            #inputs, targets = inputs.to(device), targets.to(device)
            input = shape_board2tensor(input,player)
            input.to(device)

            optimizer.zero_grad() # 勾配をゼロクリア

            outputs = model(inputs) # 順伝播 (-1から1の連続値が出力される)

            # targetsは-1から1の連続値 (float型)
            # outputsの形状: (B, 1)
            # targetsの形状: (B, 1)
            loss = criterion(outputs, targets) # 損失計算

            loss.backward() # 逆伝播
            optimizer.step() # パラメータ更新

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), 'model_weight_value.pth')

"""
# --- 使用例と訓練のセットアップ ---

# デバイスの選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ネットワークの入力とAdaptive Poolingの出力の空間形状を定義
input_shape = (9, 9, 81, 20)
output_spatial_shape = (9, 9, 81) # Adaptive Pooling後の空間形状

resnet_layers_config = [3, 4, 6, 3]
num_classes_for_resnet = 1000

# Custom3DNetScalarOutputのインスタンス化
model = Custom3DNetScalarOutput(
    input_shape=input_shape,
    output_spatial_shape=output_spatial_shape,
    resnet_layers=resnet_layers_config,
    num_classes_for_resnet=num_classes_for_resnet
).to(device)

print("\n--- モデルのサマリー ---")
summary(model, input_size=input_shape, device=str(device))
print("------------------------\n")
"""

"""
# --- ダミーデータの作成 ---
batch_size = 2
num_samples = 10

# ダミー入力データ (形状: batch_size, channels, depth, width, height)
dummy_inputs = torch.randn(num_samples, *input_shape)

# ダミー正解データ (形状: batch_size, 1) - -1から1の連続値 (float型)
# 例としてランダムな値を生成
dummy_targets = 2 * torch.rand(num_samples, 1, dtype=torch.float) - 1 # -1から1の範囲

# DataLoaderの作成
dataset = TensorDataset(dummy_inputs, dummy_targets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 損失関数と最適化手法
criterion = nn.MSELoss() # 回帰タスクのためMSELossを使用
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--- モデル訓練開始 ---")
train_model(model, train_loader, criterion, optimizer, num_epochs=5, device=device)
print("--- モデル訓練終了 ---")

# --- 訓練後の推論と出力形状の確認 ---
print("\n--- 訓練後のモデルによる推論と出力形状の確認 ---")
model.eval()
with torch.no_grad():
    sample_input = torch.randn(1, *input_shape).to(device)
    output_scalar = model(sample_input)
    print(f"推論結果 (スカラー値) の形状: {output_scalar.shape}")
    print(f"推論結果 (スカラー値): {output_scalar.item():.4f}")

print("\nこれで、モデルの構成が `resblock → Conv3d → BatchNorm → ReLU → AdaptiveAvgPool3d → Flatten → Dense → Dense → Tanh` となり、最終出力が `-1~1` の範囲のスカラー値となるバージョンが完成しました。")
"""
