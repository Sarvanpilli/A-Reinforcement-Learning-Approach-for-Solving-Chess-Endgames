from PIL import Image, ImageDraw

def create_chessboard():
    # 8x8 board, 64px per square -> 512x512
    cell_size = 64
    board_size = cell_size * 8
    img = Image.new('RGB', (board_size, board_size), 'white')
    draw = ImageDraw.Draw(img)
    
    colors = ['#F0D9B5', '#B58863'] # Light and Dark squares
    
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
    img.save('chessboard.gif')
    print("Created chessboard.gif")

if __name__ == "__main__":
    create_chessboard()
