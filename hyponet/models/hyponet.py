import torch
import torch.nn as nn
import pywt
import numpy as np
import math
from .xresnet1d import xresnet1d50


def create_head1d(
    nf: int,  # 杈撳叆鐗瑰緛鏁?
    nc: int,  # 杈撳嚭绫诲埆鏁?
    act="relu",  # 婵€娲诲嚱鏁扮被鍨嬶細ReLU 鎴?ELU
    concat_pooling=True  # 鏄惁浣跨敤杩炴帴姹犲寲
):
    """
    鐢熸垚涓€涓畝鍖栫増鐨勬ā鍨嬪ご閮紝鍖呮嫭姹犲寲灞傘€佸睍骞冲眰銆佺嚎鎬у眰锛堝彲閫変腑闂村眰鍜屾縺娲诲嚱鏁帮級銆?

    鍙傛暟锛?
    - nf: 杈撳叆鐗瑰緛鏁?
    - nc: 杈撳嚭绫诲埆鏁?
    - lin_ftrs: 涓棿灞傜壒寰佹暟锛岄粯璁や负 None
    - ps: Dropout 姒傜巼锛岄粯璁や负 0.5
    - act: 婵€娲诲嚱鏁扮被鍨嬶紙"relu" 鎴?"elu"锛?
    - concat_pooling: 鏄惁浣跨敤杩炴帴姹犲寲锛岄粯璁や负 True
    """

    # 榛樿鎯呭喌涓嬭缃?lin_ftrs锛屽鏋滄病鏈夋彁渚涘垯璁句负 [2*nf, nc] 鎴?[nf, nc]
    lin_ftrs = [2 * nf , nc]
    # 婵€娲诲嚱鏁?
    actn = nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)
    # 鍒涘缓涓€涓ā鍧楀垪琛ㄦ潵淇濆瓨缃戠粶灞?
    layers = []
    # 娣诲姞姹犲寲灞傦細濡傛灉浣跨敤杩炴帴姹犲寲锛屽垯浣跨敤 AdaptiveAvgPool1d锛涘惁鍒欎娇鐢?MaxPool1d銆?
    if concat_pooling:
        layers.append(nn.AdaptiveAvgPool1d(1))
    # 娣诲姞灞曞钩灞傦紝灏嗘暟鎹睍骞充负涓€缁?
    layers.append(nn.Flatten())
    # 杩斿洖涓€涓『搴忕殑妯″瀷
    return nn.Sequential(*layers)

class PositionalEncoding:
    @staticmethod
    def get_positional_encoding(pe_type, num_patches, d_model, learn_pe=True):
        """
        鍒涘缓浣嶇疆缂栫爜
        pe_type: 浣嶇疆缂栫爜绫诲瀷 ['zeros', 'sin_cos', 'lin1d', 'lin2d', 'exp1d', 'exp2d']
        num_patches: patch鐨勬暟閲?
        d_model: 缂栫爜缁村害
        """
        if pe_type == 'zeros':
            # 绠€鍗曠殑鍙涔犱綅缃紪鐮?
            pe = torch.zeros((num_patches, d_model))
            nn.init.uniform_(pe, -0.02, 0.02)

        elif pe_type == 'sin_cos':
            # Transformer椋庢牸鐨剆in-cos浣嶇疆缂栫爜
            pe = torch.zeros(num_patches, d_model)
            position = torch.arange(0, num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)

        elif pe_type == 'lin1d':
            # 涓€缁寸嚎鎬т綅缃紪鐮?
            pe = (2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1)) - 1)
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)

        elif pe_type == 'exp1d':
            # 涓€缁存寚鏁颁綅缃紪鐮?
            pe = (2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1) ** 0.5) - 1)
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)

        elif pe_type == 'lin2d':
            # 浜岀淮绾挎€т綅缃紪鐮?
            pe = 2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1)) * (
                torch.linspace(0, 1, d_model).reshape(1, -1)) - 1
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)

        elif pe_type == 'exp2d':
            # 浜岀淮鎸囨暟浣嶇疆缂栫爜
            pe = 2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1) ** 0.5) * (
                        torch.linspace(0, 1, d_model).reshape(1, -1) ** 0.5) - 1
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)

        else:
            raise ValueError(f"涓嶆敮鎸佺殑浣嶇疆缂栫爜绫诲瀷: {pe_type}")

        return nn.Parameter(pe, requires_grad=learn_pe)


class SignalPatcherWithPE(nn.Module):
    def __init__(self, patch_len, stride, d_model, pe_type='sin_cos', padding_patch='end', learn_pe=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Patch灞?
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.padding_patch = padding_patch

        # 鎶曞奖灞傦細灏唒atch_len缁村害鏄犲皠鍒癲_model
        self.projection = nn.Linear(patch_len, d_model)

        # 璁＄畻patch鏁伴噺
        # 濡傛灉鏈塸adding锛宲atch鏁伴噺浼氬鍔?
        self.num_patches = None  # 灏嗗湪forward涓牴鎹疄闄呰緭鍏ヨ绠?

        # 浣嶇疆缂栫爜绫诲瀷
        self.pe_type = pe_type
        self.learn_pe = learn_pe
        self.W_pos = None  # 灏嗗湪forward涓垵濮嬪寲

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 杈撳叆 x: [batch_size, channels, seq_len]

        # 1. Patch鎿嶄綔
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)

        # 浣跨敤unfold鎿嶄綔灏嗕俊鍙峰垎鎴恜atches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [batch_size, channels, num_patches, patch_len]

        # 璋冩暣缁村害椤哄簭
        x = x.permute(0, 1, 3, 2)
        # x: [batch_size, channels, patch_len, num_patches]

        # 2. 鑾峰彇鎴栧垵濮嬪寲浣嶇疆缂栫爜
        if self.W_pos is None:
            self.num_patches = x.size(-1)
            self.W_pos = PositionalEncoding.get_positional_encoding(
                self.pe_type, self.num_patches, self.d_model, self.learn_pe
            ).to(x.device)

        # 3. 鎶曞奖鍒癲_model缁村害
        B, C, P, L = x.shape
        x = x.permute(0, 1, 3, 2)  # [B, C, L, P]
        x = self.projection(x)  # [B, C, L, d_model]

        # 4. 娣诲姞浣嶇疆缂栫爜
        x = x + self.W_pos

        # 5. Dropout
        x = self.dropout(x)

        return x



class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scaling = self.d_k ** -0.5

        # 涓讳俊鍙凤紙绗竴缁村害锛夌殑娉ㄦ剰鍔涙姇褰?
        self.primary_q = nn.Linear(d_model, d_model)
        self.primary_k = nn.Linear(d_model, d_model)
        self.primary_v = nn.Linear(d_model, d_model)

        # 杈呭姪淇″彿鐨勬敞鎰忓姏鎶曞奖
        self.auxiliary_k = nn.Linear(d_model, d_model)
        self.auxiliary_v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, primary_signal, auxiliary_signals):
        """
        primary_signal: [batch, num_patches, d_model] - 绗竴缁村害鐨勪俊鍙?
        auxiliary_signals: [batch, 2, num_patches, d_model] - 鍏朵粬涓や釜缁村害鐨勪俊鍙?
        """
        B, L, D = primary_signal.shape

        # 澶勭悊涓讳俊鍙?
        Q = self.primary_q(primary_signal).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K_primary = self.primary_k(primary_signal).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V_primary = self.primary_v(primary_signal).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 澶勭悊杈呭姪淇″彿
        K_aux = self.auxiliary_k(auxiliary_signals).view(B, 2, L, self.n_heads, self.d_k).transpose(2, 3)
        V_aux = self.auxiliary_v(auxiliary_signals).view(B, 2, L, self.n_heads, self.d_k).transpose(2, 3)

        # 鍚堝苟涓讳俊鍙峰拰杈呭姪淇″彿鐨凨鍜孷
        K = torch.cat([K_primary.unsqueeze(1), K_aux], dim=1)  # [B, 3, n_heads, L, d_k]
        V = torch.cat([V_primary.unsqueeze(1), V_aux], dim=1)  # [B, 3, n_heads, L, d_k]

        # 璁＄畻娉ㄦ剰鍔涘垎鏁?
        K = K.mean(dim=1)  # 鍚堝苟鎵€鏈変俊鍙风殑K
        V = V.mean(dim=1)  # 鍚堝苟鎵€鏈変俊鍙风殑V

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 璁＄畻杈撳嚭
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out(out)

        return out


import torch
import torch.nn as nn

# 瀹氫箟涓€涓皝瑁?Patch銆佷俊鍙峰垎绂诲拰浜ゅ弶娉ㄦ剰鍔涚殑妯″潡
class PatchCrossBlock(nn.Module):
    def __init__(self, patcher, cross_attention):
        super().__init__()
        self.patcher = patcher
        self.cross_attention = cross_attention

    def forward(self, x):
        # 1. Patch 鍜屼綅缃紪鐮?
        patches = self.patcher(x)  # [batch, 3, num_patches, d_model]
        # 2. 鍒嗙涓讳俊鍙峰拰杈呭姪淇″彿
        primary_signal = patches[:, 0]    # [batch, num_patches, d_model]
        auxiliary_signals = patches[:, 1:]  # [batch, 2, num_patches, d_model]
        # 3. 浜ゅ弶娉ㄦ剰鍔?
        attended = self.cross_attention(primary_signal, auxiliary_signals)
        #浣跨敤res
        attended = attended + primary_signal
        return attended


import torch
import torch.nn as nn
import pywt
import numpy as np


# 鎻愬彇鍑?wavelet_packet_transform 鍑芥暟
def wavelet_packet_transform(x, wavelet, wp_level=3):
    """
    瀵硅緭鍏ヤ俊鍙疯繘琛屽皬娉㈠寘鍒嗚В
    x: [batch, channels, seq_len]
    wavelet: 灏忔尝鍩哄嚱鏁板悕绉?
    wp_level: 灏忔尝鍖呭垎瑙ｇ殑灞傛暟
    """
    batch_size, channels, seq_len = x.shape
    device = x.device

    # 鍒濆鍖栬緭鍑虹壒寰?
    features = []

    # 瀵规瘡涓牱鏈拰姣忎釜閫氶亾杩涜澶勭悊
    for b in range(batch_size):
        channel_features = []
        for c in range(channels):
            # 鑾峰彇褰撳墠淇″彿
            signal = x[b, c].detach().cpu().numpy()  # 鍏堣浆鎹负 numpy 杩涜澶勭悊

            # 杩涜灏忔尝鍖呭垎瑙?
            wp = pywt.WaveletPacket(signal, wavelet=wavelet, mode='symmetric', maxlevel=wp_level)

            # 鑾峰彇鎵€鏈夎妭鐐?
            nodes = [node.path for node in wp.get_level(wp_level, 'natural')]

            # 鎻愬彇姣忎釜鑺傜偣鐨勭郴鏁?
            coeffs = np.array([wp[node].data for node in nodes])

            # 鏈€灏忛暱搴﹀～鍏咃紝纭繚鎵€鏈夎妭鐐圭郴鏁伴暱搴︿竴鑷?
            min_len = min(len(coef) for coef in coeffs)
            coeffs = np.array([coef[:min_len] for coef in coeffs])

            # 杞崲鍥?Tensor 骞舵坊鍔犲埌閫氶亾鐗瑰緛鍒楄〃
            channel_features.append(torch.from_numpy(coeffs).float())

        # 鎷兼帴鎵€鏈夐€氶亾鐨勭壒寰?[num_nodes, min_len]
        channel_features = torch.cat(channel_features, dim=0)
        features.append(channel_features.unsqueeze(0))

    # 鎷兼帴鎵€鏈夋壒娆＄殑鐗瑰緛 [batch, num_nodes, min_len]
    features = torch.cat(features, dim=0).to(device)
    return features


def infer_wavelet_packet_shape(seq_len, input_channels, wavelet, wp_level):
    """Return the channel and coefficient dimensions produced by WPT."""
    dummy_signal = np.zeros(seq_len, dtype=np.float32)
    wp = pywt.WaveletPacket(dummy_signal, wavelet=wavelet, mode='symmetric', maxlevel=wp_level)
    nodes = [node.path for node in wp.get_level(wp_level, 'natural')]
    coeff_len = min(len(wp[node].data) for node in nodes)
    return input_channels * len(nodes), coeff_len


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_channels=24,  # 杈撳叆淇″彿鐨勯€氶亾鏁帮紙鐗瑰緛缁村害锛夛紝渚嬪 24
                 seq_len=756,  # 杈撳叆搴忓垪闀垮害锛屼緥濡?756
                 d_model=64,  # Transformer宓屽叆缁村害锛堣緭鍑虹壒寰佺淮搴︼級
                 nhead=8,  # 澶氬ご娉ㄦ剰鍔涚殑澶存暟
                 num_layers=4,  # Transformer Encoder 灞傛暟
                 dim_feedforward=512,  # 鍓嶅悜浼犳挱闅愯棌灞傜淮搴︼紙FFN鐨勭淮搴︼級
                 dropout=0.1,  # dropout姣斾緥
                 patch_size=6,  # 鏃跺簭鍒嗗潡澶у皬
                 use_sin_pos_enc=True  # 鏄惁浣跨敤鍥哄畾姝ｅ鸡浣嶇疆缂栫爜锛堝惁鍒欎娇鐢ㄥ彲瀛︿範浣嶇疆缂栫爜锛?
                 ):
        super(TimeSeriesTransformer, self).__init__()
        self.patch_size = patch_size

        # 1. 鏃堕棿搴忓垪鍒嗗潡宓屽叆锛圥atch Embedding锛夛細閫氳繃 Conv1d 瀹炵幇锛岄檷浣庡簭鍒楅暱搴﹀苟鎶曞奖鍒癳mbedding缁村害
        # 浣跨敤鍗风Н灏嗙浉閭荤殑 patch_size 涓椂闂存闀垮帇缂╀负涓€涓?token 鍚戦噺
        self.patch_embed = nn.Conv1d(in_channels=input_channels,
                                     out_channels=d_model,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        # 璁＄畻缁忚繃鍒嗗潡宓屽叆鍚庣殑搴忓垪闀垮害
        # 鑻eq_len涓嶈兘琚玴atch_size鏁撮櫎锛屾垜浠皢鍦╢orward涓搴忓垪杩涜閫傚綋鐨刾adding
        self.out_seq_len = math.ceil(seq_len / patch_size)  # 鐩爣杈撳嚭鐨則oken鏁伴噺锛堜緥濡?128锛?

        # 2. 浣嶇疆缂栫爜锛氫负搴忓垪涓殑姣忎釜浣嶇疆娣诲姞浣嶇疆淇℃伅
        if use_sin_pos_enc:
            # 鍥哄畾姝ｅ鸡浣嶇疆缂栫爜
            position = torch.arange(0, self.out_seq_len).unsqueeze(1).float()  # [out_seq_len, 1]
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pos_embed = torch.zeros(self.out_seq_len, d_model)
            pos_embed[:, 0::2] = torch.sin(position * div_term)  # 鍋舵暟缁村害浣跨敤sin
            pos_embed[:, 1::2] = torch.cos(position * div_term)  # 濂囨暟缁村害浣跨敤cos
            pos_embed = pos_embed.unsqueeze(0)  # [1, out_seq_len, d_model]
            self.register_buffer("pos_encoding", pos_embed)
        else:
            # 鍙涔犵殑浣嶇疆缂栫爜
            self.pos_encoding = nn.Parameter(torch.zeros(1, self.out_seq_len, d_model))

        # 3. Transformer Encoder锛氬爢鍙犲灞傝嚜娉ㄦ剰鍔涘拰鍓嶉缃戠粶
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 鍙€夛細瀵规渶鍚庤緭鍑鸿繘琛孡ayerNorm锛堟彁鍗囪缁冪ǔ瀹氭€э級
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        杈撳叆: x锛屽紶閲忕淮搴?[batch, input_channels, seq_len]
        杈撳嚭: 鐗瑰緛寮犻噺 [batch, d_model, out_seq_len]
        """
        # 纭繚鏃堕棿搴忓垪闀垮害鑳借patch_size鏁撮櫎锛屼笉瓒充箣澶勭敤0濉厖
        L = x.shape[-1]
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x = F.pad(x, (0, pad_len))  # 鍦ㄦ渶鍚庢椂闂存鏂瑰悜杩涜padding

        # 1. 鍒嗗潡宓屽叆锛氶€氳繃鍗风Н瀹炵幇patch embedding
        # 杈撳叆 x: [batch, input_channels, seq_len]
        # 杈撳嚭 x_embed: [batch, d_model, out_seq_len]
        x_embed = self.patch_embed(x)

        # 2. 璋冩暣寮犻噺缁村害椤哄簭浠ラ€傞厤 Transformer 杈撳叆锛氬彉涓?[batch, out_seq_len, d_model]
        x_embed = x_embed.permute(0, 2, 1)

        # 3. 娣诲姞浣嶇疆缂栫爜
        if hasattr(self, "pos_encoding"):
            # 鎴彇鎴栨墿鍏呬綅缃紪鐮佸埌褰撳墠搴忓垪闀垮害锛堜竴鑸笌out_seq_len鐩哥瓑锛?
            pos_enc = self.pos_encoding[:, :x_embed.size(1), :]
            x_embed = x_embed + pos_enc

        # 4. 搴旂敤 Dropout锛堝鏋滄湁锛?
        # 锛堝湪 TransformerEncoderLayer 鍐呴儴涔熸湁 Dropout锛岃繖閲屼富瑕侀拡瀵逛綅缃紪鐮佺浉鍔犲悗鐨勭粨鏋滐級
        x_embed = F.dropout(x_embed, p=0.1, training=self.training)

        # 5. Transformer Encoder 鍓嶅悜浼犳挱
        # 杈撳叆 [batch, out_seq_len, d_model] 杈撳嚭 [batch, out_seq_len, d_model]
        x_transformed = self.transformer_encoder(x_embed)

        # 6. 锛堝彲閫夛級褰掍竴鍖栬緭鍑?
        x_transformed = self.norm(x_transformed)

        # 7. 灏嗚緭鍑鸿浆缃洖 [batch, d_model, out_seq_len] 褰㈠紡锛屼互鍖归厤 SignalProcessor 鍚庣画澶勭悊闇€姹?
        x_out = x_transformed.permute(0, 2, 1)
        return x_out


class SignalProcessor(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model, n_heads=8, dropout=0.,
                 num_classes=1, wp_level=3, input_channels=3):
        super().__init__()
        # 娣诲姞褰掍竴鍖栧眰
        self.input_norm = nn.BatchNorm1d(input_channels)  # 瀵硅緭鍏ラ€氶亾杩涜褰掍竴鍖?
        # Patch鍜屼綅缃紪鐮佸眰
        self.patcher = SignalPatcherWithPE(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            pe_type='sin_cos',
            padding_patch='end'
        )

        # 浜ゅ弶娉ㄦ剰鍔涘眰
        self.cross_attention = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # 灏?Patch銆佷俊鍙峰垎绂汇€佷氦鍙夋敞鎰忓姏灏佽鍒颁竴涓ā鍧椾腑
        self.patch_cross = PatchCrossBlock(self.patcher, self.cross_attention)
        self.wp_level = wp_level  # 灏忔尝鍖呭垎瑙ｇ骇鍒?
        # 涓烘畫宸儴鍒嗛€夋嫨db4灏忔尝鍩猴紙閫傚悎澶勭悊鍣０鍜屽揩閫熷彉鍖栫殑淇″彿锛?
        self.res_wavelet = 'db4'
        # 涓鸿秼鍔块儴鍒嗛€夋嫨sym5灏忔尝鍩猴紙閫傚悎澶勭悊骞虫粦瓒嬪娍淇″彿锛?
        self.trend_wavelet = 'db2'
        res_wp_channels, res_wp_len = infer_wavelet_packet_shape(
            seq_len, input_channels, self.res_wavelet, self.wp_level
        )
        trend_wp_channels, trend_wp_len = infer_wavelet_packet_shape(
            seq_len, input_channels, self.trend_wavelet, self.wp_level
        )
        # 绗竴涓壒寰佹彁鍙栨ā鍧?
        self.wp_res_feature = TimeSeriesTransformer(input_channels=res_wp_channels, seq_len=res_wp_len, d_model=64, nhead=8, num_layers=4,dim_feedforward=512, dropout=0.1, patch_size=6)
        # 绗簩涓壒寰佹彁鍙栨ā鍧?
        self.wp_trend_feature = TimeSeriesTransformer(input_channels=trend_wp_channels, seq_len=trend_wp_len, d_model=64, nhead=8, num_layers=4,dim_feedforward=512, dropout=0.1, patch_size=6)

        # 璁＄畻patch鏁伴噺
        self.num_patches = (seq_len - patch_len) // stride + 1
        if patch_len == 'end':
            self.num_patches += 1

        # 鐗瑰緛铻嶅悎灞?
        self.resnet = xresnet1d50(input_channels=187 , num_classes=1, dropout=0.)

        # Handcrafted physiological features: 81 -> 128.
        self.mlp = nn.Sequential(
            nn.Linear(81, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.head = create_head1d(
            32,
            nc=num_classes,
            act="relu",
            concat_pooling=True,
        )

        # Fusion head: Patch-Conv/time (128) + DWT-Former/frequency (128)
        # + handcrafted-feature MLP (128) = 384.
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x, ml_feature=None):
        """
        x: [batch, 3, seq_len]
        """
        # 鍏堣繘琛屽綊涓€鍖?
        x = self.input_norm(x)
        #visualize_signals_and_wavelets(x[0,:,:], db4_level=3, db2_level=3)

        #灏忔尝鐗瑰緛鍒濆鍖?
        wavelet_features = None

        # 瀵规畫宸拰瓒嬪娍鍒嗗埆杩涜灏忔尝鍖呭彉鎹?
        res_wp_features = wavelet_packet_transform(x, self.res_wavelet, wp_level=self.wp_level)
        trend_wp_features = wavelet_packet_transform(x, self.trend_wavelet, wp_level=self.wp_level)

        # 浣跨敤transformer缃戠粶鎻愬彇wavelet_features鐗瑰緛銆?
        res_features = self.wp_res_feature(res_wp_features)
        trend_features = self.wp_trend_feature(trend_wp_features)

        # 灞曞钩鐗瑰緛
        res_features = res_features.max(dim=-1)[0]
        trend_features = trend_features.max(dim=-1)[0]
        # 鍚堝苟鐗瑰緛
        wavelet_features = torch.cat((res_features, trend_features), dim=1)
        #浣跨敤灏佽濂界殑 patch_cross 妯″潡锛岃繖閲屾槸Patch-conv妯″潡鎻愬彇net_feature
        attended = self.patch_cross(x)
        # 鐗瑰緛铻嶅悎
        net_feature= self.resnet(attended)
        output = self.head(net_feature)
        output = torch.cat((output,wavelet_features),dim=1)
        if self.classifier.in_features == output.shape[1] + 128:
            if ml_feature is None:
                raise ValueError("ml_feature is required for this SignalProcessor checkpoint.")
            ml_features = self.mlp(ml_feature)
            output = torch.cat((output, ml_features), dim=1)
        elif self.classifier.in_features != output.shape[1]:
            raise ValueError(
                f"Unexpected fusion dimension: classifier expects {self.classifier.in_features}, "
                f"but available features have {output.shape[1]} dimensions."
            )
        # 鍒嗙被
        output = self.classifier(output)

        return output
