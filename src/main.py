import numpy as np
import scipy.fftpack as fftpack

data = np.array([
  [ 52,  55,  61,  66,  70,  61,  64,  73],
  [ 63,  59,  66,  90, 109,  85,  69,  72],
  [ 62,  59,  68, 113, 144, 104,  66,  73],
  [ 63,  58,  71, 122, 154, 106,  70,  69],
  [ 67,  61,  68, 104, 126,  88,  68,  70],
  [ 79,  65,  60,  70,  77,  68,  58,  75],
  [ 85,  71,  64,  59,  55,  61,  65,  83],
  [ 87,  79,  69,  68,  65,  76,  78,  94]
])

# First, we''ll have to center the 8x8 block
# our range for each number becomes [-128,127]
for i in range(8):
  for j in range(8):
    data[i][j] -= 128

# check
print(data)

# reason we shifted is because DCT (discrete cosine transform works best with values centered around 0)

# now, we must use the 2d DCT
rows_results = []
for row in data:
  transformed = fftpack.dct(row, norm='ortho')
  rows_results.append(transformed)

print(rows_results)

horizontal_dict = np.stack(rows_results) # Now turn into matrix (8x8) again
# apply dct to that now
horizontal_results = fftpack.dct(horizontal_dict, norm='ortho', axis=0) # provide axis for cols instead of rows

print(horizontal_results)

Q_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]) # kindly provided by gemini

dct_matrix = np.array(horizontal_results)

quantized = np.round(dct_matrix / Q_TABLE).astype(int)

print(quantized)

ZIG_ZAG_INDICES = [
    (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2), (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
    (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2), (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
    (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4), (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
    (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6), (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
] # gemini again

zigzag_stream = [quantized[r][c] for r, c in ZIG_ZAG_INDICES]

print(zigzag_stream)

last_idx = 0
for i in range(len(zigzag_stream) - 1, -1, -1):
    if zigzag_stream[i] != 0:
        last_idx = i
        break
compressed_data = zigzag_stream[:last_idx + 1]
print(f"Comprssed stream: {compressed_data} + [EOB]")

# i'm tired, i'll continue later.
