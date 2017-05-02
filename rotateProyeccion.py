import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import matplotlib as plt
from matplotlib import pyplot as plt


#proyeccion horizontal (imagen de entrada=binaria)
def proyeccion_v(img,grados):
    filas,columnas= img.shape
    M = cv2.getRotationMatrix2D((columnas/2,filas/2),grados,1)
    mask_blanca=cv2.bitwise_xor(img,cv2.bitwise_not(img))
    dst_mask = cv2.warpAffine(mask_blanca,M,(columnas,filas))
    ret, mask = cv2.threshold(dst_mask,190,255,cv2.THRESH_BINARY)
    dst = cv2.warpAffine(img,M,(columnas,filas))
    ret, rot_thresh = cv2.threshold(dst,190,255,cv2.THRESH_BINARY)
    dst=cv2.bitwise_not(cv2.bitwise_xor(rot_thresh,mask))
    #cv2.namedWindow('imag', cv2.WINDOW_NORMAL)
    #cv2.imshow('imag',dst)
    #cv2.waitKey(0)
    proy_v=[]
    sum=0
    for i in range(0, columnas):
       for j in range(0,filas):
                if dst[j][i]==0:
                    sum=sum+1
       proy_v.append(sum)
       sum=0
    #print(proy_v)
    #plt.plot(proy_v)
    #plt.show()
    return proy_v


#calcula las proyecciones verticales
#retorna una lista con las proyecciones verticales
def calcular_proyecciones_v(brazo1,brazo2,sentido):
    proyecciones_b1=[]
    proyecciones_b2=[]
    b1_fila,b1_columna=brazo1.shape
    b2_fila,b2_columna=brazo2.shape
    #print(sentido)
    if sentido: #si True-> #girar en sentido horario el brazo
        for i in range(-5,13):
            ang_i=-i*5
            proy_v=proyeccion_v(brazo1,ang_i)
            proyecciones_b1.append((proy_v,ang_i,np.count_nonzero(proy_v)))
        for i in range(-5,13):
            ang_i=i*5
            proy_v=proyeccion_v(brazo2,ang_i)
            proyecciones_b2.append((proy_v,ang_i,np.count_nonzero(proy_v)))
    else:  #si False-> #girar en sentido horario el brazo superior
        for i in range(-5,13): #braso superior
            ang_i=i*5
            proy_v=proyeccion_v(brazo1,ang_i)
            proyecciones_b1.append((proy_v,ang_i,np.count_nonzero(proy_v)))
        for i in range(-5,13):
            ang_i=-i*5
            proy_v=proyeccion_v(brazo2,ang_i)
            proyecciones_b2.append((proy_v,ang_i,np.count_nonzero(proy_v)))

    return proyecciones_b1,proyecciones_b2

def mejor_proyeccion_v(p_b1,p_b2):
    valores=[]

    #brazo1
    for i in range(0,len(p_b1)):
        valores.append(p_b1[i][2])
    import operator
    min_index, min_value = min(enumerate(valores), key=operator.itemgetter(1))
    ang1=p_b1[min_index][1]

    #brazo2
    valores=[]
    for i in range(0,len(p_b2)):
        valores.append(p_b2[i][2])
    min_index, min_value = min(enumerate(valores), key=operator.itemgetter(1))
    ang2=p_b2[min_index][1]
    return ang1,ang2

def eliminar_objetos_pequeños(th,tamano):
    blobs_labels = measure.label(th, background=255)
    boxes_objetos= ndimage.find_objects(blobs_labels)
    #print(boxes_objetos[0][0])
    #ejemplo1
    #(slice(0, 2, None), slice(19, 23, None))
    #So this object is found at x = 19–23 and y = 0–2
    if len(boxes_objetos)>1:#solo debe haber 1 objeto->el cromosoma
        #filtro objetos muy pequeños ya que sob objetos indeseados (no cromosomas)
        aux_boxes=[]
        for l in range(0,len(boxes_objetos)):
            if len(np.transpose(np.nonzero(cv2.bitwise_not(th[boxes_objetos[l]]))))<tamano:
                th[boxes_objetos[l]]=255
    return th


def floodfill(th):
    # Copy the thresholded image.
    im_floodfill = th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = th | im_floodfill_inv

    return im_out


#----------------------------------
#----------------------------------
def proyeccion_h2(gray,grados):
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    th=cv2.bitwise_not(floodfill(cv2.bitwise_not(th))) #rellenar huecos (si quedo alguno)
    th=eliminar_objetos_pequeños(th,20) # si quedo algun pixel suelto




    #cv2.namedWindow('th', cv2.WINDOW_NORMAL)
    #cv2.imshow('th',th)
    #cv2.waitKey(0)


    filas,columnas= th.shape
    mask_blanca=cv2.bitwise_xor(th,cv2.bitwise_not(th))
    M = cv2.getRotationMatrix2D((columnas/2,filas/2),grados,1)
    dst = cv2.warpAffine(th,M,(columnas,filas))
    gray_rot=cv2.warpAffine(gray,M,(columnas,filas))
    dst_mask = cv2.warpAffine(mask_blanca,M,(columnas,filas))
    ret, rot_thresh = cv2.threshold(dst,190,255,cv2.THRESH_BINARY)
    ret, mask = cv2.threshold(dst_mask,190,255,cv2.THRESH_BINARY)

    #ret, rot_thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.namedWindow('th rotado', cv2.WINDOW_NORMAL)
    #cv2.imshow('th rotado',rot_thresh)
    #cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    #cv2.imshow('mask',mask)
    dst=cv2.bitwise_not(cv2.bitwise_xor(rot_thresh,mask))
    dst_gray=cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(dst),cv2.bitwise_not(gray_rot)))
    #cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    #cv2.imshow('dst',dst_gray)
    #cv2.waitKey(0)
    proy_h=[]
    sum=0
    for i in range(0, filas):
           for j in range(0,columnas):
                    if dst[i][j]==0: #cuento si es pixel negro
                        sum=sum+1
           proy_h.append(sum)
           sum=0
    return proy_h,dst,dst_gray


#calcula las proyecciones horizontales entre ... y ... grados
#retorna una lista con las proyecciones horizontales
def calcular_proyecciones(gray):
    proyecciones=[]
    for i in range(-25,25): # con paso de 5 grados
        ang_i=i*5
        proy_h,dst,dst_gray=proyeccion_h2(gray,ang_i)
        #cv2.namedWindow('imagen'+str(i), cv2.WINDOW_NORMAL)
        #cv2.imshow('imagen'+str(i),dst_gray)
        #cv2.waitKey(0)
        proyecciones.append((proy_h,ang_i,dst,dst_gray))
    return proyecciones


def analisis2(proyecciones):
    vector_min=[]
    S=[]
    #print(proyecciones)
    for i in range(0,len(proyecciones)):
        proy_h_i=proyecciones[i][0]
        angulo=proyecciones[i][1]
        #cv2.namedWindow('imagen'+str(i), cv2.WINDOW_NORMAL)
        #cv2.imshow('imagen'+str(i),dst_gray)
        #cv2.waitKey(0)
        #contar ceros
        count=0
        for j in range(0,len(proy_h_i)):
            if proy_h_i[j]==0:
                count=count+1
            else:
                break
        offset=count+6
        count=0
        proy_h_i_offset=proy_h_i[offset:len(proy_h_i)-offset]
        if len(proy_h_i_offset)==0:
            offset=count+2
            proy_h_i_offset=proy_h_i[offset:len(proy_h_i)-offset]
        minimo=proy_h_i_offset[0]
        index_min=0
        for kk in range(0,len(proy_h_i_offset)):
            if minimo>proy_h_i_offset[kk]:
                minimo=proy_h_i_offset[kk]
                index_min=kk
        if index_min!=0 and index_min!=len(proy_h_i_offset)-1: #si pasa esto no girar la imagen, no hacer nada
            #partir
            izquierda=proy_h_i_offset[0:index_min]
            derecha=proy_h_i_offset[index_min:len(proy_h_i_offset)-1]
            #print(izquierda)
            #print(derecha)

            #izquierda
            maximo_izq=izquierda[0]
            index_izq=0
            for ii in range(0,len(izquierda)):
                if maximo_izq<izquierda[ii]:
                    maximo_izq=izquierda[ii]
                    index_izq=ii

            #derecha
            maximo_der=derecha[0]
            index_der=0
            for ii in range(0,len(derecha)):
                if maximo_der<derecha[ii]:
                    maximo_der=derecha[ii]
                    index_der=ii
            index_der=index_der+index_min


            if index_izq!=0 and index_izq < index_min and index_min< index_der and index_der!=len(proy_h_i_offset)-1:
                R1=abs(proy_h_i_offset[index_izq]-proy_h_i_offset[index_der])/(proy_h_i_offset[index_izq]+proy_h_i_offset[index_der])
                R2=proy_h_i_offset[index_min]/(proy_h_i_offset[index_izq]+proy_h_i_offset[index_der])
                S_i=0.42*R1+0.54*R2
                #print(S_i)
                S.append((S_i,angulo,offset,index_izq,index_min,index_der))

    if len(S)!=0: #
        #print(S)
        S_min=S[0][0]
        S_index=0
        for i in range(0,len(S)):
            if S_min>S[i][0]:
                S_min=S[i][0]
                S_index=i
        #print(S_index)
        for i in range(0,len(proyecciones)):
            if S[S_index][1]==proyecciones[i][1]:
                dst=proyecciones[i][2]
                dst_gray=proyecciones[i][3]
            #cv2.namedWindow('imagen'+str(i), cv2.WINDOW_NORMAL)
            #cv2.imshow('imagen'+str(i),dst)
            #cv2.waitKey(0)
        return S[S_index],dst,dst_gray
    else:#si vacio considero el mejor ajuste de la imagen el actual, por lo tanto no la modifico. le dejo =
        return -1,-1,-1#si S==0 dejo la imagen como esta!

def sentido_giro(dst,index_min,offset,filas,columnas):
    aux=np.zeros((filas,columnas),np.uint8)
    aux=cv2.bitwise_not(aux)
    cv2.line(aux,(0,index_min+offset),(columnas,index_min+offset),0,1)
    aux=cv2.bitwise_not(aux)
    aux2=cv2.bitwise_and(aux,dst)#marco la linea
    aux3=aux2[index_min+offset:index_min+offset+1, 0:columnas] #recorto exactamente la linea
    no_zero_elem= np.nonzero(cv2.bitwise_not(aux3))
    posicion=(no_zero_elem[1][0]+no_zero_elem[1][len(no_zero_elem[1])-1])/2
    #print(no_zero_elem)
    #print(posicion)
    #print(columnas/2)
    #cv2.imshow("aux",aux3)
    #cv2.waitKey(0)
    if posicion>(columnas/2): #girar en sentido horario el brazo superior, antihorario el inferior
        return True
    else: #girar en sentido horario el  brazo inferior, antihorario el superior
        return False

def girar_pegar_brazos(brazo1,brazo2,brazo1_gray,brazo2_gray,ang1,ang2):
    #brazo1
    filas,columnas= brazo1.shape
    mask_blanca=cv2.bitwise_xor(brazo1,cv2.bitwise_not(brazo1))
    M = cv2.getRotationMatrix2D((columnas/2,filas/2),ang1,1)
    dst = cv2.warpAffine(brazo1,M,(columnas,filas))
    gray_rot=cv2.warpAffine(brazo1_gray,M,(columnas,filas))
    dst_mask = cv2.warpAffine(mask_blanca,M,(columnas,filas))
    ret, rot_thresh = cv2.threshold(dst,254,255,cv2.THRESH_BINARY)
    ret, mask = cv2.threshold(dst_mask,254,255,cv2.THRESH_BINARY)
    dst=cv2.bitwise_not(cv2.bitwise_xor(rot_thresh,mask))
    dst_gray_brazo_1=cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(dst),cv2.bitwise_not(gray_rot)))
    #brazo2
    filas2,columnas2= brazo2.shape
    mask_blanca=cv2.bitwise_xor(brazo2,cv2.bitwise_not(brazo2))
    M = cv2.getRotationMatrix2D((columnas2/2,filas2/2),ang2,1)
    dst = cv2.warpAffine(brazo2,M,(columnas2,filas2))
    gray_rot=cv2.warpAffine(brazo2_gray,M,(columnas2,filas2))
    dst_mask = cv2.warpAffine(mask_blanca,M,(columnas2,filas2))
    ret, rot_thresh = cv2.threshold(dst,254,255,cv2.THRESH_BINARY)
    ret, mask = cv2.threshold(dst_mask,254,255,cv2.THRESH_BINARY)
    dst=cv2.bitwise_not(cv2.bitwise_xor(rot_thresh,mask))
    dst_gray_brazo_2=cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(dst),cv2.bitwise_not(gray_rot)))

    #proyecciones verticales de los brazos
    #brazo1
    suma_acum=[]
    for ii in range(0,filas):
        suma=0
        for jj in range(0,columnas):
            if dst_gray_brazo_1[ii][jj]!=255:
                suma=suma+dst_gray_brazo_1[ii][jj]
        suma_acum.append(suma)
    for ii in reversed(range(0,len(suma_acum)-1)):
        if suma_acum[ii]!=0:
            index=ii
            break
    dst_gray_brazo_1= dst_gray_brazo_1[0:index, 0:columnas]
    #brazo2
    suma_acum=[]
    for ii in range(0,filas2):
        suma=0
        for jj in range(0,columnas2):
            if dst_gray_brazo_2[ii][jj]!=255:
                suma=suma+dst_gray_brazo_2[ii][jj]
        suma_acum.append(suma)
    for ii in range(0,len(suma_acum)-1):
        if suma_acum[ii]!=0:
            index=ii
            break
    dst_gray_brazo_2= dst_gray_brazo_2[index:filas2, 0:columnas2]
    #determinar cuantas columnas mover para pegar los brazos
    fila1,col1=dst_gray_brazo_1.shape
    count1=0
    for i in range(0,col1):
        if dst_gray_brazo_1[fila1-1][i]!=255:
            count1=count1+1
            index1=i
    fila2,col2=dst_gray_brazo_2.shape
    count2=0
    for i in range(0,col2):
        if dst_gray_brazo_2[0][i]!=255:
            index2=i
            count2=count2+1

    #print("----------")
    #print(index1)
    #print(count1)
    #print(index2)
    #print(count2)
    a=int(round(count1/2))
    b=int(round(count2/2))
    index11=index1-a
    index22=index2-b
    offset=abs(int(round((index22-index11))))
    if index11<=index22:#el de arriba(index1) esta mas a la izquierda(termina antes) que index2->la muevo a la derecha


        dst_gray_brazo_1=cv2.copyMakeBorder(dst_gray_brazo_1, top=0, bottom=fila2, left=offset, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )
        dst_gray_brazo_1=dst_gray_brazo_1[0:fila1+fila2,0:col1]
        dst_gray_brazo_2=cv2.copyMakeBorder(dst_gray_brazo_2, top=fila1-2, bottom=2, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )

    else: #index1>index2 --> el de abajo(index2) esta mas a la izquierda(termina antes) que index1

        dst_gray_brazo_1=cv2.copyMakeBorder(dst_gray_brazo_1, top=0, bottom=fila2, left=0, right=offset, borderType= cv2.BORDER_CONSTANT, value=255 )
        dst_gray_brazo_1=dst_gray_brazo_1[0:fila1+fila2,offset:col1+offset]
        dst_gray_brazo_2=cv2.copyMakeBorder(dst_gray_brazo_2, top=fila1-2, bottom=2, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )


    #print(dst_gray_brazo_1.shape)
    #print(dst_gray_brazo_2.shape)

    nueva=cv2.bitwise_not(cv2.add(cv2.bitwise_not(dst_gray_brazo_1),cv2.bitwise_not(dst_gray_brazo_2)))
    #cv2.namedWindow('aux2', cv2.WINDOW_NORMAL)
    #cv2.imshow("aux2", nueva)
    #cv2.waitKey(0)
    return nueva




def cortarImagen(S,dst,dst_gray,gray,perimetro_normalizado):
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    #ret, th = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    #th= cv2.morphologyEx(th, cv2.MORPH_CLOSE,kernel)#elimino huecos
    #th= cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)#elimino ruido del contorno del cromosoma
    th=cv2.bitwise_not(floodfill(cv2.bitwise_not(th))) #rellenar huecos (si quedo alguno)
    th=eliminar_objetos_pequeños(th,20)
    #cv2.namedWindow('aux2', cv2.WINDOW_NORMAL)
    #cv2.imshow("aux2", th)
    #cv2.waitKey(0)
    img, contours, _ = cv2.findContours(cv2.bitwise_not(th), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try: #si fitEllipse no obtine 5 puntos de contorno error-> no se porque no encuentra los 5 puntos requeridos, entonces cologo este try
        ellipse = cv2.fitEllipse(contours[0])
        cv2.ellipse(th,ellipse,0,1)
        #print('--')
        #print(len(contours[0]))
        (x,y),(MA,ma),angs = cv2.fitEllipse(contours[0])
        #print(angs)
    except:
        return gray
    #- Si vacio considero el mejor ajuste de la imagen el actual->no la modifico,
    #- Si al angulo del elipse esta cerca a 180 la considero vertical -> no modifico
    #- Si en cromosoma es pequeño-> en general no esta curvo -> no nodifico ( para determinar que es pequeño
    #primero determnino el largo del eje medio de todos los cromosomas de la celula de caso de estudio y
    #y si el lasgo es menor al 60% del mas largo lo considero pequeño--> perimetro del contorno
    #al incluir esta script en el helloimage
    #vector S de func analisis2 (S_i,angulo,offset,index_izq,index_min,index_der)
    if   S!=-1 and perimetro_normalizado>=0.17: #5<angs<176 and
        #if S[1]>=10 and S[1]<=150:
        offset=S[2]
        index_min=S[4]
        filas,columnas=dst_gray.shape
        brazo1= dst[0:index_min+offset, 0:columnas] # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        brazo2= dst[index_min+offset:filas, 0:columnas]
        #cv2.imshow("cropped", brazo2)
        #cv2.waitKey(0)
        #determinar como debo girar, si el paso por la linea se da a la derecha o a izquierda de la mitad
        sentido=sentido_giro(dst,index_min,offset,filas,columnas)
        #print("aca1")
        brazo1=cv2.copyMakeBorder(brazo1, top=0, bottom=index_min+offset, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )
        brazo2=cv2.copyMakeBorder(brazo2, top=index_min+offset, bottom=0, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )
        p_b1,p_b2=calcular_proyecciones_v(brazo1,brazo2,sentido)
        ang1,ang2=mejor_proyeccion_v(p_b1,p_b2)
        #giro los brasos en la imagen de gris
        brazo1_gray= dst_gray[0:index_min+offset, 0:columnas]
        brazo2_gray= dst_gray[index_min+offset:filas, 0:columnas]
        brazo1_gray=cv2.copyMakeBorder(brazo1_gray, top=0, bottom=index_min+offset, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )
        brazo2_gray=cv2.copyMakeBorder(brazo2_gray, top=index_min+offset, bottom=0, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255 )
        nueva=girar_pegar_brazos(brazo1,brazo2,brazo1_gray,brazo2_gray,ang1,ang2)

        #verificacion
        #antes de retornar la imagen "nueva" verifico que la proyeccion vertical de la nueva sea "menos ancha"
        #que la de la original -> sino retorno la original
        proy_v_nueva=[]
        filas, columnas=nueva.shape
        sum_nueva=0
        for i in range(0, columnas):
           sum_nueva=0
           for j in range(0,filas):
                    if nueva[j][i]!=255:
                        sum_nueva=sum_nueva+1
           proy_v_nueva.append(sum_nueva)
        nueva_no_zero=np.count_nonzero(proy_v_nueva)

        proy_v=[]
        filas, columnas=gray.shape
        for i in range(0, columnas):
           sum=0
           for j in range(0,filas):
                    if gray[j][i]!=255:
                        sum=sum+1
           proy_v.append(sum)
        orig_no_zero=np.count_nonzero(proy_v)
        if nueva_no_zero<orig_no_zero: #retorno la de proyeccion vertical mas angosta
            return nueva
        else:
            return gray
    else:
        return gray

'''

# load the image from disk
for i in range(0,46):
    img = cv2.imread('/home/asusn56/PycharmProjects/ejemploWatershed/single/97002138.9.tiff/'+str(i)+'_97002138.9.tiff')
    #img = cv2.imread('/home/asusn56/PycharmProjects/ejemploWatershed/single/9700TEST.7.tiff/7_9700TEST.7.tiff')
    cv2.namedWindow('imagen', cv2.WINDOW_NORMAL)
    cv2.imshow("imagen",img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    filas,columnas=gray.shape
    if filas>columnas:
            bordersize=int(round(filas/2))
    else:
            bordersize=input(round(columnas/2))
    gray=cv2.copyMakeBorder(gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )

    #proyeccion_h2(gray,90)
    proyecciones=calcular_proyecciones(gray)
    S,dst,dst_gray=analisis2(proyecciones)
    dst1=cortarImagen(S,dst,dst_gray,gray)
    cv2.imwrite('/home/asusn56/PycharmProjects/RorateCorretly/rotate/'+str(i)+'.tiff',dst1)


cv2.waitKey(0)

'''
