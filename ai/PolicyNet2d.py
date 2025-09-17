import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary # torchsummary がない場合は pip install torchsummary でインストール
import torch.nn.functional as F # LogSoftmax のために必要

# CircularConv2dの定義 (3Dから2Dへ変更)
class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CircularConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        if isinstance(padding, int):
            self.pad = (padding, padding, padding, padding) # W_start, W_end, H_start, H_end
        elif isinstance(padding, tuple) and len(padding) == 2: # (pH, pW)
            self.pad = (padding[1], padding[1], padding[0], padding[0]) # W_start, W_end, H_start, H_end
        elif isinstance(padding, tuple) and len(padding) == 4: # (W_s, W_e, H_s, H_e)
            self.pad = padding
        else:
            raise ValueError("Padding must be int or tuple of 2 or 4 ints.")

    def forward(self, x):
        x = F.pad(x, self.pad, mode='circular')
        return self.conv(x)

# 修正されたResNetDynamicAlpha.pyをインポートすることを確認してください。
from ResNetDynamicAlpha import get_CustomResNet, Resblock

class Custom2DNetSpatialOutput(nn.Module): # クラス名をCustom2DNetSpatialOutputに変更
    def __init__(self,
                 input_shape_original_3d: tuple, # 元の3D入力形状 (channels, depth, width, height)
                 output_spatial_shape_2d: tuple, # 目的の出力の2D空間形状 (height, width)
                 resnet_layers: list,
                 num_classes_for_resnet: int):
        """
        カスタム2Dニューラルネットワーク。
        ResNet -> Conv2d -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d -> Conv2d -> LogSoftmax の構成。
        元の3D入力を2Dに変換して処理する。
        最終的な出力は (batch_size, output_height, output_width) となり、
        各 (output_height, output_width) の要素の総和が1 (のログ) となる確率分布を表す。

        Args:
            input_shape_original_3d (tuple): 入力データの元の3D形状 (channels, depth, width, height)。
                                            Goボードは width x height と解釈され、channels x depth は特徴量に統合される。
            output_spatial_shape_2d (tuple): 目的の出力データの2D空間形状 (height, width)。
            resnet_layers (list): ResNetの各ステージのブロック数を定義するリスト。
            num_classes_for_resnet (int): ResNetモデルの初期化に必要だが、ここでは特徴抽出器としてのみ使用される。
        """
        super(Custom2DNetSpatialOutput, self).__init__()

        # 元の3D入力形状を保存 (リシェイプのために使用)
        self.original_in_channels, self.original_in_depth, self.original_in_width, self.original_in_height = input_shape_original_3d

        # 2D ResNetへの入力形状を計算
        # Goボードの空間次元は original_in_width x original_in_height
        # 新しい2D入力チャネル数は original_in_channels * original_in_depth
        self.resnet_input_channels_2d = self.original_in_channels * self.original_in_depth
        self.resnet_input_spatial_shape_2d = (self.original_in_width, self.original_in_height) # (height, width)

        # ResNetに渡すための2D入力形状タプル
        resnet_input_shape_2d = (self.resnet_input_channels_2d, *self.resnet_input_spatial_shape_2d)

        self.out_height, self.out_width = output_spatial_shape_2d # 最終出力の空間形状

        self.output_channels = 1 # ポリシーネットワークの最終出力チャネル数 (通常は1)

        # ResNet部分 - CircularConv2dを使用するように修正されたもの
        self.resnet = get_CustomResNet(
            block=Resblock,
            layers=resnet_layers,
            num_classes=num_classes_for_resnet, # ダミー値
            input_shape=resnet_input_shape_2d # 2D入力形状を渡す
        )

        # ResNetからの出力形状を推論 (バッチサイズ, ResNet出力チャネル, H, W)
        dummy_resnet_input = torch.randn(1, *resnet_input_shape_2d)
        with torch.no_grad():
            dummy_resnet_output = self.resnet.forward_features(dummy_resnet_input)

        resnet_output_channels = dummy_resnet_output.shape[1]

        # ResNetからの特徴マップを処理する中間層
        # CircularConv2dを使用
        self.conv2d = CircularConv2d(in_channels=resnet_output_channels,
                                out_channels=64, # 中間チャネル数
                                kernel_size=(3, 3),
                                padding=(1, 1)) # kernel_size=3なのでpadding=1

        self.bn2d = nn.BatchNorm2d(64) # BatchNorm3d -> BatchNorm2d
        self.relu = nn.ReLU()

        # 出力空間形状に合わせるためのAdaptive Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.out_height, self.out_width)) # AdaptiveAvgPool3d -> AdaptiveAvgPool2d

        # 最終的な出力チャネルを1にするための畳み込み層
        # CircularConv2dを使用
        self.final_conv = CircularConv2d(in_channels=64,
                                    out_channels=self.output_channels,
                                    kernel_size=(1, 1), # 1x1 conv
                                    padding=0) # 1x1 convなのでpaddingは0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 元の3D入力 (B, C, D, W, H) を 2D入力 (B, C*D, W, H) にリシェイプ
        batch_size = x.shape[0]
        # x_2d = x.view(batch_size, self.resnet_input_channels_2d, self.original_in_width, self.original_in_height)
        # より汎用的なリシェイプ (C_in, D_in, W_in, H_in) -> (C_in * D_in, W_in, H_in)
        x_2d = x.view(batch_size, self.original_in_channels * self.original_in_depth, self.original_in_width, self.original_in_height)


        # ResNetで特徴抽出 (2D入力)
        resnet_features = self.resnet.forward_features(x_2d)

        # 中間畳み込み層 (2D)
        x = self.conv2d(resnet_features)
        x = self.bn2d(x)
        x = self.relu(x)

        # Adaptive Poolingで出力形状に調整 (2D)
        x = self.adaptive_pool(x)

        # 最終畳み込み層 (2D)
        x = self.final_conv(x)

        # 出力形状の調整とLogSoftmaxの適用
        # 出力は (batch_size, channels=1, height, width)
        # LogSoftmaxを適用するために、(height, width) を一つの次元にフラット化
        x_reshaped_for_softmax = x.view(batch_size, -1) # (B, H*W)

        # LogSoftmaxを適用 (全ピクセルに対して確率分布)
        x_log_probs_flat = F.log_softmax(x_reshaped_for_softmax, dim=-1)
        # NaN/Infを防ぐための処理: 出力テンソルにNaNやInfが含まれないようにする
        x = torch.nan_to_num(x, nan=0.0, posinf=torch.finfo(output.dtype).max, neginf=torch.finfo(output.dtype).min)

        # 元の空間形状 (H, W) に戻す
        x_final = x_log_probs_flat.view(batch_size, self.out_height, self.out_width)

        return x_final


# --- 訓練用ヘルパー関数 ---
def train_model_policy(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train() # モデルを訓練モードに設定
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets,player in dataloader:
            inputs, targets = torch.tensor(inputs).float().to(device), torch.tensor(targets).float().to(device)

            optimizer.zero_grad() # 勾配をゼロにリセット

            outputs = model(inputs) # フォワードパス
            loss = criterion(outputs, targets) # 損失を計算
            print(torch.isnan(outputs).any()) # outputsにNaNがあるか
            print(torch.isinf(outputs).any()) # outputsにinfがあるか
            print(torch.isnan(targets).any()) # targetsにNaNがあるか
            print(torch.isinf(targets).any()) # targetsにinfがあるか
            loss.backward() # バックプロパゲーション
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # パラメータを更新

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

# --- モデルのインスタンス化と訓練の例 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 入力と出力の形状定義
    # 元の3D入力形状 (channels, depth, width, height)
    # Goボードの空間は width x height (例: 19x19)
    # channelsとdepthは特徴量チャネルに統合
    original_input_shape_3d = (1,16, 9, 9) # (channels, depth, width, height) として全て19x19x19の例
                                               # Goボードは19x19と仮定し、最初の2つの19は特徴量チャネルに統合
    output_spatial_shape_2d = (9, 9) # ポリシー出力の2D空間形状 (height, width)

    # ResNetの設定
    resnet_layers_config = [3, 4, 6, 3] # ResNet50相当の層構成
    num_classes_for_resnet = 1000 # ResNetの最終FC層はここでは使用しないが、定義のために必要

    # モデルのインスタンス化
    model = Custom2DNetSpatialOutput( # クラス名を変更
        input_shape_original_3d=original_input_shape_3d,
        output_spatial_shape_2d=output_spatial_shape_2d,
        resnet_layers=resnet_layers_config,
        num_classes_for_resnet=num_classes_for_resnet
    ).to(device)

    # summary関数に渡す入力形状は、モデルのforwardメソッドが期待する元の3D入力形状
    print("\n--- モデルのサマリー ---")
    summary(model, input_size=original_input_shape_3d, device=str(device))
    print("------------------------\n")


    # --- ダミーデータの作成 ---
    batch_size = 2
    num_samples = 10 # 訓練用のダミーサンプル数

    # ダミー入力データ (形状: batch_size, original_in_channels, original_in_depth, original_in_width, original_in_height)
    dummy_inputs = torch.randn(num_samples, *original_input_shape_3d)

    # ダミー正解データ (ポリシー出力は確率分布なので、LogSoftmaxのターゲットはログ確率)
    # 形状: batch_size, output_height, output_width
    dummy_targets = torch.empty(num_samples, *output_spatial_shape_2d)
    for i in range(num_samples):
        # (H, W) の2D平面で確率分布を生成
        probs_hw = torch.rand(*output_spatial_shape_2d)
        probs_hw_normalized = probs_hw / probs_hw.sum()
        log_probs_hw = torch.log(probs_hw_normalized + 1e-10) # ログ確率に変換
        dummy_targets[i, :, :] = log_probs_hw

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
print("\n--- 訓練後のモデルによる推論と出力形状の確認 ---\n")
model.eval() # モデルを評価モードに設定
with torch.no_grad():
    sample_input = torch.randn(1, *original_input_shape_3d).to(device)
    output_log_probabilities = model(sample_input)
    print(f"推論結果 (確率の対数) の形状: {output_log_probabilities.shape}")

    # 出力全体の総和が1になるか確認 (ログ確率なのでexpして総和)
    test_probs = torch.exp(output_log_probabilities[0]) # 最初のサンプルの確率
    print(f"出力の確率の総和 (期待値: 1.0): {test_probs.sum().item():.4f}")
