import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary # torchsummary がない場合は pip install torchsummary でインストール
import torch.nn.functional as F

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

class Custom2DNetScalarOutput(nn.Module): # クラス名をCustom2DNetScalarOutputに変更
    def __init__(self,
                 input_shape_original_3d: tuple, # 元の3D入力形状 (channels, depth, width, height)
                 output_spatial_shape_2d: tuple, # Adaptive Poolingの目的の出力2D空間形状 (height, width)
                 resnet_layers: list,
                 num_classes_for_resnet: int):
        """
        カスタム2Dニューラルネットワーク。
        ResNet -> Conv2d -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d -> Flatten -> Dense -> Dense -> Tanh の構成。
        元の3D入力を2Dに変換して処理する。
        最終的な出力は -1から1 のスカラー値となる。

        Args:
            input_shape_original_3d (tuple): 入力データの元の3D形状 (channels, depth, width, height)。
                                            Goボードは width x height と解釈され、channels x depth は特徴量に統合される。
            output_spatial_shape_2d (tuple): AdaptiveAvgPool2d の目的の出力2D空間形状 (height, width)。
            resnet_layers (list): ResNetの各ステージのブロック数を定義するリスト。
            num_classes_for_resnet (int): ResNetモデルの初期化に必要だが、ここでは特徴抽出器としてのみ使用される。
        """
        super(Custom2DNetScalarOutput, self).__init__()

        # 元の3D入力形状を保存 (リシェイプのために使用)
        self.original_in_channels, self.original_in_depth, self.original_in_width, self.original_in_height = input_shape_original_3d

        # 2D ResNetへの入力形状を計算
        self.resnet_input_channels_2d = self.original_in_channels * self.original_in_depth
        self.resnet_input_spatial_shape_2d = (self.original_in_width, self.original_in_height) # (height, width)

        # ResNetに渡すための2D入力形状タプル
        resnet_input_shape_2d = (self.resnet_input_channels_2d, *self.resnet_input_spatial_shape_2d)

        self.out_height, self.out_width = output_spatial_shape_2d

        self.output_dimension = 1 # バリューネットワークの出力次元 (通常は1: 勝率や評価値)

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

        # Adaptive Poolingで出力空間形状を固定 (2D)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.out_height, self.out_width)) # AdaptiveAvgPool3d -> AdaptiveAvgPool2d

        # Flatten後の特徴量サイズを計算
        flattened_features = 64 * self.out_height * self.out_width

        # 全結合層
        self.fc1 = nn.Linear(flattened_features, 512)
        self.fc2 = nn.Linear(512, self.output_dimension)
        self.tanh = nn.Tanh() # 出力を -1から1 の範囲に制限するため

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 元の3D入力 (B, C, D, W, H) を 2D入力 (B, C*D, W, H) にリシェイプ
        batch_size = x.shape[0]
        x_2d = x.view(batch_size, self.original_in_channels * self.original_in_depth, self.original_in_width, self.original_in_height)

        # ResNetで特徴抽出 (2D入力)
        resnet_features = self.resnet.forward_features(x_2d)

        # 中間畳み込み層 (2D)
        x = self.conv2d(resnet_features)
        x = self.bn2d(x)
        x = self.relu(x)

        # Adaptive Poolingで空間形状を固定 (2D)
        x = self.adaptive_pool(x)

        # Flattenして全結合層へ
        x = x.view(x.size(0), -1)

        # 全結合層と活性化関数
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x) # 出力を-1から1に正規化 (勝敗や評価値に適応)
        x = torch.nan_to_num(x, nan=0.0)

        return x


# --- 訓練用ヘルパー関数 ---
def train_model_value(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train() # モデルを訓練モードに設定
    for epoch in range(num_epochs):
        running_loss = 0.0
        target = train_loader[0]
        target = target.to(device)
        del dataloader[0]
        for input, _, player in dataloader:
            inputs, targets = torch.tensor(inputs).float().to(device), torch.tensor(targets).float().to(device)

            optimizer.zero_grad() # 勾配をゼロにリセット

            outputs = model(inputs) # フォワードパス
            loss = criterion(outputs, targets) # 損失を計算
            loss.backward() # バックプロパゲーション
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # パラメータを更新

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

"""
# --- モデルのインスタンス化と訓練の例 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 入力と出力の形状定義
    # 元の3D入力形状 (channels, depth, width, height)
    original_input_shape_3d = (1,8, 19, 19) # Goボードは19x19と仮定し、最初の2つの19は特徴量チャネルに統合
    output_spatial_shape_2d = (1, 1) # Adaptive Poolingの最終空間形状 (height, width)

    # ResNetの設定
    resnet_layers_config = [3, 4, 6, 3] # ResNet50相当の層構成
    num_classes_for_resnet = 1000 # ResNetの最終FC層はここでは使用しないが、定義のために必要

    # モデルのインスタンス化
    model = Custom2DNetScalarOutput( # クラス名を変更
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

    # ダミー正解データ (形状: batch_size, 1) - -1から1の連続値 (float型)
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
    print("\n--- 訓練後のモデルによる推論と出力形状の確認 ---\n")
    model.eval() # モデルを評価モードに設定
    with torch.no_grad():
        sample_input = torch.randn(1, *original_input_shape_3d).to(device)
        output_value = model(sample_input)
        print(f"推論結果 (評価値) の形状: {output_value.shape}")
        print(f"推論結果のサンプル値: {output_value.item():.4f}")
"""
