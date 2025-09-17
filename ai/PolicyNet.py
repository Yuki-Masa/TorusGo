import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary # torchsummary がない場合は pip install torchsummary でインストール
import torch.nn.functional as F # LogSoftmax のために必要

# ResNetDynamicAlpha.py は同じディレクトリにあると仮定します。
from ResNetDynamicAlpha import get_CustomResNet, Resblock

class Custom3DNetSpatialOutput(nn.Module):
    def __init__(self,
                 input_shape: tuple,  # (channels, depth, width, height)
                 output_spatial_shape: tuple, # (depth, width, height) - 出力の空間形状
                 resnet_layers: list, # ResNetの各ステージのブロック数を定義するリスト (例: [3, 4, 6, 3])
                 num_classes_for_resnet: int):
        """
        カスタム3Dニューラルネットワーク。
        ResNet -> Conv3d -> BatchNorm3d -> ReLU -> AdaptiveAvgPool3d -> Conv3d -> LogSoftmax の構成。
        最終的な出力は (batch_size, output_depth, output_width, output_height) となり、
        特定の height 位置における (depth, width) の要素の総和が1 (のログ) となる確率分布を表す。

        Args:
            input_shape (tuple): 入力データの形状 (channels, depth, width, height)。
            output_spatial_shape (tuple): 目的の出力データの空間形状 (depth, width, height)。
            resnet_layers (list): get_CustomResNet関数に渡す、各ステージのブロック数を定義するリスト。
            num_classes_for_resnet (int): ResNetモデルの初期化に必要ですが、
                                         このネットワークではResNetの最終FC層は使用しません。
        """
        super(Custom3DNetSpatialOutput, self).__init__()

        # 入力形状の展開
        in_channels, in_depth, in_width, in_height = input_shape
        # 出力空間形状の展開
        self.out_depth, self.out_width, self.out_height = output_spatial_shape

        # 最終出力のチャネル数は1
        self.output_channels = 1

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
        self.relu = nn.ReLU()

        # Adaptive poolingで出力の空間次元をoutput_spatial_shapeに調整
        self.adaptive_pool = nn.AdaptiveAvgPool3d((self.out_depth, self.out_width, self.out_height))

        # 最終的な出力チャネルを1にするためのConv3d (LogSoftmaxへのロジット)
        self.final_conv = nn.Conv3d(in_channels=64,
                                    out_channels=self.output_channels, # 出力チャネルは常に1
                                    kernel_size=(1, 1, 1))

        # LogSoftmax層はforwardメソッド内でF.log_softmaxを使って適用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ネットワークの順伝播。

        Args:
            x (torch.Tensor): 入力テンソル (batch_size, channels, depth, width, height)。

        Returns:
            torch.Tensor: 出力テンソル (batch_size, depth, width, height) の形式で、
                          各 height スライスにおいて (depth, width) の総和が1 (のログ) となる確率の対数。
        """
        # ResNetによる特徴抽出
        resnet_features = self.resnet.forward_features(x)

        # Conv3d -> BatchNorm3d -> ReLU
        x = self.conv3d(resnet_features)
        x = self.bn3d(x)
        x = self.relu(x)

        # Adaptive poolingで空間次元を目的の出力サイズに合わせる
        x = self.adaptive_pool(x)

        # 最終チャネル数を1に調整 (ロジットの出力): (B, 1, D, W, H)
        x = self.final_conv(x)

        # LogSoftmax を (depth, width) 次元に適用するためにテンソルを再整形
        # (B, 1, D, W, H) を (B, H, D*W) に変換し、Softmax適用後、元の形状に戻す

        # 1. height 次元を分離 (permute)
        # (B, 1, D, W, H) -> (B, H, 1, D, W)
        x_permuted = x.permute(0, 4, 1, 2, 3)

        batch_size, H_out, C_dummy, D_out, W_out = x_permuted.shape

        # 2. (depth, width) 次元を平坦化 (reshape)
        # (B, H, 1, D, W) -> (B, H, D*W)
        # 各 H スライスごとに D*W の次元を Softmax の対象にする
        x_reshaped_for_softmax = x_permuted.reshape(batch_size, H_out, C_dummy * D_out * W_out)

        # 3. LogSoftmax を平坦化された次元 (dim=-1) に適用
        # 出力は確率の対数
        x_log_probs_flat = F.log_softmax(x_reshaped_for_softmax, dim=-1)

        # 4. 結果を元の形状に戻す
        # (B, H, D*W) -> (B, H, 1, D, W) -> (B, 1, D, W, H)
        x_reshaped_back = x_log_probs_flat.view(batch_size, H_out, C_dummy, D_out, W_out)
        x_final = x_reshaped_back.permute(0, 2, 3, 4, 1)

        # 5. チャネル次元が1なので削除し、(B, D, W, H) の形にする
        x_final = x_final.squeeze(1)

        return x_final

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
        for inputs, targets, player in train_loader:
            inputs = shape_board2tensor(inputs,player)
            target = rotate_board(target)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad() # 勾配をゼロクリア

            outputs = model(inputs) # 順伝播 (LogSoftmax適用後の確率の対数が出力される)

            # targetsはLogSoftmax適用後の確率分布 (B, D, W, H)
            # outputsの形状: (B, D, W, H)
            # targetsの形状: (B, D, W, H)
            loss = criterion(outputs, targets) # 損失計算
            print(torch.isnan(outputs).any()) # outputsにNaNがあるか
            print(torch.isinf(outputs).any()) # outputsにinfがあるか
            print(torch.isnan(targets).any()) # targetsにNaNがあるか
            print(torch.isinf(targets).any()) # targetsにinfがあるか

            loss.backward() # 逆伝播
            optimizer.step() # パラメータ更新

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), 'model_weight_policy.pth')

"""
# --- 使用例と訓練のセットアップ ---

# デバイスの選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ネットワークの入力と出力の形状を定義
input_shape = (9, 9, 81, 20)
output_spatial_shape = (9, 9, 81)

resnet_layers_config = [3, 4, 6, 3]
num_classes_for_resnet = 1000

# Custom3DNetSpatialOutputのインスタンス化
model = Custom3DNetSpatialOutput(
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

# ダミー正解データ (形状: batch_size, depth, width, height) - 各 (D,W) スライスが確率分布
dummy_targets = torch.empty(num_samples, *output_spatial_shape, dtype=torch.float)

# 各 height スライスにおいて (depth, width) の総和が1となる確率分布を生成し、ログを取る
for i in range(num_samples):
    for h_idx in range(output_spatial_shape[2]): # height 次元をループ
        # (D, W) のロジットを生成し、Softmaxを適用して確率分布にする
        logits_dw = torch.randn(output_spatial_shape[0], output_spatial_shape[1])
        # (D*W) に平坦化してLogSoftmaxを適用し、元の形状に戻す
        probs_dw_flat = F.softmax(logits_dw.view(-1), dim=-1)
        log_probs_dw = torch.log(probs_dw_flat + 1e-10).view(output_spatial_shape[0], output_spatial_shape[1])
        dummy_targets[i, :, :, h_idx] = log_probs_dw


# DataLoaderの作成
dataset = TensorDataset(dummy_inputs, dummy_targets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 損失関数と最適化手法
criterion = nn.KLDivLoss(reduction='batchmean') # ログ確率間のKLダイバージェンスを計算
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--- モデル訓練開始 ---")
train_model(model, train_loader, criterion, optimizer, num_epochs=5, device=device)
print("--- モデル訓練終了 ---")

# --- 訓練後の推論と出力形状の確認 ---
print("\n--- 訓練後のモデルによる推論と出力形状の確認 ---")
model.eval()
with torch.no_grad():
    sample_input = torch.randn(1, *input_shape).to(device)
    output_log_probabilities = model(sample_input)
    print(f"推論結果 (確率の対数) の形状: {output_log_probabilities.shape}")

    # 特定の height スライスでの (depth, width) の総和が1になるか確認
    # 例: 最初のサンプル、最初のheightスライス (インデックス0)
    test_slice_log_probs = output_log_probabilities[0, :, :, 0] # (D, W)
    test_slice_probs = torch.exp(test_slice_log_probs)
    print(f"最初のサンプル、最初のheightスライス (確率):\n{test_slice_probs}")
    print(f"最初のサンプル、最初のheightスライス (確率の総和): {test_slice_probs.sum():.4f}")

    # 確率値を取得したい場合
    output_probabilities = torch.exp(output_log_probabilities)
    print(f"推論結果 (確率) の形状: {output_probabilities.shape}")

print("\nこれで、特定の height 位置における (depth, width) のすべての要素の総和が1 (のログ) となるようにSoftmaxを適用したモデルを、`nn.KLDivLoss` を用いて学習させることができます。")
"""
