"""集中管理可由使用者調整的辨識與顯示參數。"""

# rembg 使用的模型；isnet-anime 適合動漫與 VTuber 角色。
DETECTION_MODEL = "isnet-anime"

# 靜態圖片第一次結果不可靠時使用的人像模型；動畫 GIF 不使用。
SECONDARY_DETECTION_MODEL = "birefnet-portrait"

# 增加留白前的裁切框達圖片 95% 時，視為不可靠並進行二次辨識。
UNRELIABLE_CROP_AREA_RATIO = 0.95

# 一般人物遮罩門檻（0～255）。
MASK_THRESHOLD = 32

# 暗場與近距離人物使用的自適應門檻。
MASK_LOW_THRESHOLD = 8
MASK_HIGH_THRESHOLD = 96

# 人物遮罩的平均亮度低於／高於這些值時，分別使用低／高門檻。
DARK_MASK_MEAN_MAX = 4.0
CLOSEUP_MASK_MEAN_MIN = 40.0

# 保留區域至少要有最大連通區域的 5%，且至少占圖片的 0.02%。
MIN_COMPONENT_RELATIVE_AREA = 0.05
MIN_COMPONENT_AREA_RATIO = 0.0002

# 相距在圖片長邊 3% 內的遮罩碎片視為同一個人物群組。
COMPONENT_JOIN_DISTANCE_RATIO = 0.03

# 接觸圖片邊界且小於圖片 1% 的區域通常是模型產生的邊界雜點。
SMALL_BORDER_COMPONENT_MAX_AREA_RATIO = 0.01

# 有效人物遮罩像素至少要占整張圖片 1%；不足時改用整張圖。
MIN_FOREGROUND_AREA_RATIO = 0.01

# 人物外接矩形四周的留白比例；0.03 代表各邊增加 3%。
DETECTION_PADDING_RATIO = 0.03

# 框外黑色遮罩透明度（0～255）；數字越大越暗。
OUTSIDE_SHADE_ALPHA = 128
