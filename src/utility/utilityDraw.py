import cv2
import numpy as np
import svgwrite
from utility.CMFD_PM_utily import dilateDisk, erodeDisk

def draw_t_text(img, text, point, quadrante, fontScale=0.4, color=(0, 0, 0), color_bk=(1, 1, 1)):
    x = img
    font = cv2.FONT_HERSHEY_PLAIN
    texts = text.split('\n')
    label_size = [cv2.getTextSize(_, font, fontScale)[0] for _ in texts]
    h = max([_[1] for _ in label_size])
    label_size = (max([_[0] for _ in label_size]), len(label_size)*h)
    print('quadrante:', quadrante)
    if quadrante==0:
        point0 = (point[0]-label_size[0], point[1])
    elif quadrante==1:
        point0 = (point[0], point[1])        
    elif quadrante==2:
        point0 = (point[0]-label_size[0], point[1]-label_size[1])
    else:
        point0 = (point[0], point[1]-label_size[1])
        
    point1 = (point0[0]+label_size[0], point0[1]+label_size[1])
    
    x = cv2.rectangle(x, np.int32(point0), np.int32(point1), color_bk, -1)
    for i,_ in enumerate(texts):
        point1 = (point0[0], point0[1]+i*h+h)
        x = cv2.putText(x, _, np.int32(point1), font, fontScale, color)

    return x

def draw_t_arrows(img, start_points, end_points, color=(1.0, 0.0, 0.0), thickness=1, tipLength=0.1, tipWidth=4):
    x = img
    for a,b in zip(start_points, end_points):
        c = (a-b) * tipLength + b
        d = (a-b) @ np.asarray([[0, -1], [1, 0]])
        
        pt0 = np.int32(np.round(a))
        pt1 = np.int32(np.round((c+b)/2))
        d = tipWidth * thickness * d / np.sqrt(np.sum(d**2)) / 2.0
        x = cv2.line(x, pt0, pt1, color, thickness)
        
        triangle_cnt = np.int32(np.round(np.array([b, c+d, c-d])))
        x = cv2.drawContours(x, [triangle_cnt], 0, color, -1)
    return x


def get_hex_color(c):
    return '#%02X%02X%02X'%c

def init_svg(w, h):
    dwg = svgwrite.Drawing(size=('%fpx'%w, '%fpx'%h), viewBox='0 0 %f %f'%(w, h))
    arrow = dwg.marker(id='arrow', insert=(10, 3), size=(10, 10), orient='auto', markerUnits='strokeWidth')
    arrow.add(dwg.path(d='M0,0 L0,6 L9,3 z', fill='#E52207'))
    dwg.defs.add(arrow)

    filtr = dwg.filter(id="solid", start=(0, 0), size=(1, 1))
    filtr.feFlood(flood_color="#FFFFFF", result="bg")
    filtr.feMerge('').feMergeNode(["bg", "SourceGraphic"])
    dwg.defs.add(filtr)
    return dwg

def draw_s_image_embnpy(dwg, img, w, h):
    import base64
    from PIL import Image
    
    from io import BytesIO
    from PIL import Image
    with BytesIO() as buf:
        Image.fromarray(img).save(buf, format='png')
        base64_bytes = base64.b64encode(bytearray(buf.getvalue())).decode()
    
    dwg.add(svgwrite.image.Image(href="data:image/png;base64,"+base64_bytes,
                  insert=('0px','0px'),
                  size=('%fpx'%w, '%fpx'%h)))


def draw_s_image_embpng(dwg, path_img, w, h):
    import base64
    with open(path_img, 'rb') as fid:
        base64_bytes = base64.b64encode(fid.read()).decode()
    dwg.add(svgwrite.image.Image(href="data:image/png;base64,"+base64_bytes,
                  insert=('0px','0px'),
                  size=('%fpx'%w, '%fpx'%h)))


def draw_s_image(dwg, path_img, w, h):
    dwg.add(svgwrite.image.Image(href=path_img,
                  insert=('0px','0px'),
                  size=('%fpx'%w, '%fpx'%h)))
    
def draw_s_text(dwg, text, point, quadrante, fontScale=8, color=(0, 0, 0)):
    texts = text.split('\n')
    text_anchor ="start" if quadrante%2==1 else 'end'
    a = dwg.text("", insert=('%fpx'%point[0], '%fpx'%point[1]), fill=get_hex_color(color),
                 font_size='%dpt'%fontScale, text_anchor=text_anchor, filter="url(#solid)")
    for i,text in enumerate(texts):
        if i==0:
            if quadrante>1:
                a.add(dwg.tspan(text, x=['%fpx'%point[0],], dy=['%dpt'%(fontScale-len(texts)*fontScale),], text_anchor=text_anchor))
            else:
                a.add(dwg.tspan(text, x=['%fpx'%point[0],], text_anchor=text_anchor))
        else:
            a.add(dwg.tspan(text, x=['%fpx'%point[0],], dy=['%dpt'%fontScale,], text_anchor=text_anchor))
        
    dwg.add(a)
    
def draw_s_arrows(dwg, start_points, end_points, color=(229, 34,  7), thickness=1):
    for a,b in zip(start_points, end_points):
        line = dwg.add(svgwrite.shapes.Line(('%fpx'%a[0], '%fpx'%a[1]),
                                            ('%fpx'%b[0], '%fpx'%b[1]), stroke=get_hex_color(color),
                                            stroke_width='%dpt'%thickness, marker_end="url(#arrow)"))





def draw_f_borders(img, mask, color=(229, 34,  7)):
    bord = dilateDisk(mask,3) & (erodeDisk(mask,3)==0)
    img[bord, 0] = color[0] / 255.0
    img[bord, 1] = color[1] / 255.0
    img[bord, 2] = color[2] / 255.0
    return img


#, color=(0.7, 0.0, 0.0), thickness=1
#img = draw_text_f(img, text, c_centro, quadrante, fontScale = 0.004*dimention)
#img = draw_arrows_f(img, c_0.transpose((1,0)), c_1.transpose((1,0)), color=color, thickness=thickness)
#print(text)