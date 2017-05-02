import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import skimage
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import data
from skimage import img_as_ubyte
import os
from skimage import io
import mahotas
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy.polynomial.polynomial as poly
from itertools import starmap, tee
from math import sqrt
from operator import mul, sub
import rotateProyeccion as proy
import os, os.path
import copy




def find_skeleton(img):
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)


    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(3,3),0)
    _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters)


def contorno(thresh):
    # Contorno con --> Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #_,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    _, contours, _ = cv2.findContours(cv2.bitwise_not(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    imagen_contorno= np.zeros(thresh.shape,np.uint8)
    cv2.drawContours(imagen_contorno, contours, -1, 255, 1)
    perimetro_contorno = cv2.arcLength(contours[0],True)#perimetro del contorno
    moments = cv2.moments(contours[0])
    area_contorno = moments['m00'] ##area del cromosoma --> igual a area = cv2.contourArea(cnt)
    return area_contorno,perimetro_contorno,imagen_contorno

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


def mediaAxis(img,thresh):
    #blur = cv2.GaussianBlur(img,(3,3),0)
    #_,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    "Convert gray images to binary images using Otsu's method"
    BW_Original = img <= thresh    # must set object region as 1, background region as 0 !
    BW_Skeleton = zhangSuen(BW_Original)
    skeleto = img_as_ubyte(BW_Skeleton) #transformo skimage to opencv image
    #aa=np.array(BW_Skeleton,dtype=np.float64)#opencv image to skimage ???
    #opencv use numpy.uint8 type to represent binary images instead scikit-image numpy.float64

    sk_puntos=np.transpose(np.nonzero(skeleto)) #y,x
     #CONTORNO DE skeleto
    _, contorno_skeleto, _ = cv2.findContours(skeleto,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    largo_del_eje_medio = cv2.arcLength(contorno_skeleto[0],True)#perimetro del contorno

    return skeleto,largo_del_eje_medio

def lineas_perpendiculares(skeleto,ventana,paso):
    sk_puntos=np.transpose(np.nonzero(skeleto)) #[fila, columna]
    #print(sk_puntos)
    img_lineas=np.zeros(skeleto.shape,np.uint8)
    filas,columnas=skeleto.shape
    puntos_perpendiculares=[]
    #print(np.transpose(np.nonzero(skeleto)))
    for i in range(ventana,len(sk_puntos)-ventana,paso):
        #print(sk_puntos[i])
        #v.x = B.x - A.x; v.y = B.y - A.y;
        centro=sk_puntos[i]
        Ax=sk_puntos[i+ventana][1]
        Ay=sk_puntos[i+ventana][0]
        Bx=sk_puntos[i-ventana][1]
        By=sk_puntos[i-ventana][0]
        v_x=Bx-Ax
        v_y=By-Ay
        mag = math.sqrt(v_x*v_x + v_y*v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        temp = v_x
        v_x = -v_y
        v_y = temp
        #C.x = B.x + v.x * length; C.y = B.y + v.y * length;
        length=9
        C_x_1 = int(round(Bx +v_x * length))
        C_y_1 = int(round(By + v_y * length))
        length=-9
        C_x_2 =int(round( Bx + v_x *length))
        C_y_2 =int(round( By + v_y *length ))

        puntos_perpendiculares.append([C_x_1, C_y_1 ,C_x_2, C_y_2,Bx,By])
        cv2.line(img_lineas,(C_x_1,C_y_1),(C_x_2,C_y_2),255,1)

    return puntos_perpendiculares,img_lineas


def proyeccion_h(th,bool,grados,bordersize):
    th=cv2.copyMakeBorder(th, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=0 )
    if bool:
        th=cv2.bitwise_not(th)
    filas,columnas= th.shape
    M = cv2.getRotationMatrix2D((columnas/2,filas/2),grados,1)
    dst = cv2.warpAffine(th,M,(columnas,filas))
    #cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    #cv2.imshow('dst',dst)
    proy_h=[]
    sum=0
    for i in range(0, filas):
           for j in range(0,columnas):
                    if dst[i][j]==0: #cuento si es pixel negro
                        sum=sum+1
           proy_h.append(sum)
           sum=0
    proy_h=proy_h[bordersize:len(proy_h)-bordersize] #elimino el borde
    return proy_h

#media  de gris sobre la imagen entera del cromosoma
def avg_density(img,thresh):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    final_im=cv2.bitwise_and(gray,gray,cv2.bitwise_not(thresh))
    density=cv2.mean(final_im,cv2.bitwise_not(thresh))[0]
    return density

def distancia_euclidea(p, q):
    return sqrt(sum(starmap(mul, (zip(*tee(starmap(sub, zip(p, q))))))))

def densityprofile(gray,puntos_perpendiculares,thresh):
    #[C_x_1, C_y_1 ,C_x_2, C_y_2,Bx,By]
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask_linea=np.zeros(gray.shape,np.uint8)
    D = []
    for i in range(0,len(puntos_perpendiculares)):
        cv2.line(mask_linea,(puntos_perpendiculares[i][0],puntos_perpendiculares[i][1]),(puntos_perpendiculares[i][2],puntos_perpendiculares[i][3]),255,1)
        masked_linea=cv2.bitwise_and(cv2.bitwise_not(thresh),mask_linea)
        gray_linea=cv2.bitwise_and(gray,masked_linea)
        n=len(np.transpose(np.nonzero(gray_linea))) #cantidad de puntos de la linea
        D_i=cv2.mean(gray_linea,mask_linea)[0] #media o promedio ?
        D.append([D_i,n])
        #clear variables
        mask_linea=np.zeros(gray.shape,np.uint8)
        gray_linea=mask_linea
        masked_linea=mask_linea
    return D # D[(promedio de gris de la linea perpendicular, ancho de la linea perpendicular(ancho en pixels))

def shapeprofile(gray,puntos_perpendiculares,thresh):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask_linea=np.zeros(gray.shape,np.uint8)
    distancia_total=0
    distancia=[]
    S=[]
    for i in range(0,len(puntos_perpendiculares)):
        cv2.line(mask_linea,(puntos_perpendiculares[i][0],puntos_perpendiculares[i][1]),(puntos_perpendiculares[i][2],puntos_perpendiculares[i][3]),255,1)
        masked_linea=cv2.bitwise_and(cv2.bitwise_not(thresh),mask_linea)
        puntos=np.transpose(np.nonzero(masked_linea))
        distancia.append(distancia_euclidea(puntos[0], puntos[-1]))
        distancia_total=distancia_total+distancia[i]
        #clear variables
        mask_linea=np.zeros(gray.shape,np.uint8)
        masked_linea=mask_linea

    mask_linea=np.zeros(gray.shape,np.uint8)
    for ii in range(0,len(puntos_perpendiculares)):
        cv2.line(mask_linea,(puntos_perpendiculares[ii][0],puntos_perpendiculares[ii][1]),(puntos_perpendiculares[ii][2],puntos_perpendiculares[ii][3]),255,1)
        masked_linea=cv2.bitwise_and(cv2.bitwise_not(thresh),mask_linea)
        gray_linea=cv2.bitwise_and(gray,masked_linea)
        puntos=np.transpose(np.nonzero(gray_linea))
        #print(puntos)
        suma=0
        for iii in range(0,len(puntos)):
            g_i=gray_linea[puntos[iii][0],puntos[iii][1]]
            suma=g_i*(distancia[ii]*distancia[ii])
        S_i=suma/(distancia_total*distancia_total)
        S.append(S_i)
        #clear variables
        mask_linea=np.zeros(gray.shape,np.uint8)
        gray_linea=mask_linea
        masked_linea=mask_linea
    return S

def marcarcentromero(puntos_perpendiculares,D):
    #print("lend D de marcarcarcentromero: "+str(len(D)))
    DD=np.transpose(D)
    import operator
    min_indexD, min_valueD = min(enumerate(DD[1]), key=operator.itemgetter(1)) #minIndexD es indice de la linea perperndicular mas fina
    #print(min_indexD)
    #point
    punto=D[min_indexD]
    #coordenadas [C_x_1, C_y_1 ,C_x_2, C_y_2,Bx,By]
    #print(puntos_perpendiculares)
    primerPunto=puntos_perpendiculares[0]
    midPoint=puntos_perpendiculares[min_indexD]
    ultimoPunto=puntos_perpendiculares[len(puntos_perpendiculares)-1]
    #print(primerPunto)
    #print(midPoint)
    #print(ultimoPunto)
    #length of CI, denoted by CI(L) Lp/(Lp + Lq) ,
    # where Lp is the length of a p-arm() mas cortos en submetacentric , acosencentric, telocentric
    #  and Lq is the length of a q-arm.
    brazo2=distancia_euclidea((midPoint[4],midPoint[5]),(ultimoPunto[4],ultimoPunto[5]))
    brazo1=distancia_euclidea((midPoint[4],midPoint[5]),(primerPunto[4],primerPunto[5]))
    #print(brazo1)
    #print(brazo2)
    #p-arm es el brazo corto, esta arriba
    #q-arm es el brazo largo esta abajo
    if brazo1<brazo2:#el cromosoma esta correcto
        p_arm=brazo1
        q_arm=brazo2
        invertido=0
    else: #el cromosoma esta girado 180 grados, el brazo corto (p-arm) esta abajo
        invertido=1
        p_arm=brazo2
        q_arm=brazo1
    if p_arm==0: #si es cero es porque es un cromosoma corto (viene del problema de eje medio no lo lo marco hasta el final en las puntaspor eso puede dar cero)->mejorar media axis
        p_arm=1
    CI=p_arm/(p_arm+q_arm)
    return CI,midPoint,min_indexD,invertido


def bandingprofile(img,puntos_perpendiculares,thresh):
    #idealized banding profile is computed by processing a density profile D(x)
    median = cv2.medianBlur(img,5)
    B=densityprofile(median,puntos_perpendiculares,thresh)
    aux=np.zeros((len(B),1),np.uint8)
    fila,col=aux.shape
    for i in range(0,fila):
        aux[i,0]=int(round(B[i][0]))
    aux2=coherence_filter(aux, sigma = 11, str_sigma = 10, blend = 0.8, iter_n = 50)
    while True:
        bandera=False
        for i in range(0,len(aux2)-1):
            aa=aux2[i]*1+aux2[i+1]*(-1)
            if aa==1:
                aux2[i]=aux2[i]-1
                bandera=True
            elif aa==-1:
                aux2[i]=aux2[i]+1
                bandera=True
        #print(bandera)
        if bandera==False:
            break
    #print(aux2)
    #retornar el perfil de bandas idealizado
     #aux2 limpia aun mas algunas ruidos que quedaron entra bandas con pixeles que difieren en +1 o menos 1
            #es requerido esto??->verlo
    return aux2

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def parsear_banda(bandas,indexs):
    parser=[]
    #print(len(indexs))
    #print(bandW)
    #print(bandas)
    parser.append(bandas[0:indexs[0]])#primer banda
    for i in range(0,len(indexs)-1): #siguientes bandas hasta la anteultima
        parser.append(bandas[indexs[i]:indexs[i+1]])
    parser.append(bandas[indexs[len(indexs)-1]:len(bandas)]) #agrego ultimo elemento de la banda
    #print(parser)
    #print(len(parser))
    return parser


def caracteristicas1(thresh,bandprofile,bandBW,img,puntos_perpendiculares,indexs,minIndexD,invertido,area):
    D=densityprofile(img,puntos_perpendiculares,thresh)#obtengo el perfil de valores de pixel original
    #print("len de D de avrdarkset: "+str(len(D)))
    aux=np.zeros((len(D),1),np.uint8)
    fila,col=aux.shape
    for i in range(0,fila): #para pasar los valores a enteros
        aux[i,0]=int(round(D[i][0]))
    aux2=[]
    for i in range(0,len(aux)): #para extrar los valores a una lista simple
        aux2.append(aux[i][0])
    #print(aux2)
    parser=parsear_banda(aux2,indexs)
    # (6) the total number of detected bands in a chromosome
    cantidad_bandas=len(parser)
    #print(parser)
    promedios=[]
    for i in range(0,len(parser)):
        promedios.append(sum(parser[i])/(float(len(parser[i]))))
    import operator
    #(1) average_max=the average pixel value of the darkest band in a chromosome,
    #(2) index_avr_max=the location of the darkest band
    index_avr_max, average_max = max(enumerate(promedios), key=operator.itemgetter(1))

    #Corto por minIndexD->indica donde la linea perpendicular es mas fina-> la supongo la posi del centromero
    #minIndexD no puede ser cero(a veces me pasa eso por el tema del media axis)
    #debo mejorar el media axis debe ser de toda la extension del cromosoma
    #print("minIndexD: "+str(minIndexD))
    #print("indexs: "+str(indexs))
    #print("cantidad de bandas: "+str(cantidad_bandas))
    aux=[]
    for i in range(0,len(bandprofile)): #para extrar los valores a una lista simple
        aux.append(bandprofile[i][0])
    bandprofile=aux
    if cantidad_bandas>2:
        #buscar si minIndexD esta en la lista de indexs si esta dejo ese y si no esta le
        #le asigno el indice mas cercano de indexs
        if minIndexD in indexs: #si esta
            indexx=indexs.index(minIndexD)#busco la posicion del indice minimo en la lista de indices
        else: #si no esta->asignar el mas cercano
            diff=[]
            for i in range(0,len(indexs)):
                diff.append(abs(indexs[i]-minIndexD))
                #print("diff:" +str(diff))
                indexx, _ = min(enumerate(diff), key=operator.itemgetter(1))
        #print(indexx)
        #print(indexs)
        if indexx==0:# indexx no puede ser cero (si es cero es por que el minimo me dio en algunas de las puntas del cromosoma)  arbitrariamente asigno  del indice de la mitad del largo de indexs
            indexx=int(round(len(indexs)/2))

        #print("indice asignado: "+str(indexx))
        #print("valor indice: "+str(indexs[indexx]))
        parte1=bandprofile[:indexs[indexx]]
        parte2=bandprofile[indexs[indexx]:]
        #print("bandprofile:"+str(bandprofile))
        #print("parte1:"+str(parte1))
        #print("parte2:"+str(parte2))

        #print("----------------------------------------------------------------")
        parser_banprofile=parsear_banda(bandprofile,indexs)
        parser_parte1=parsear_banda(parte1,indexs[:indexx])
        parser_parte2=parser_banprofile[len(parser_parte1):]
        #print(parser_banprofile)
        #print(parser_parte1)
        #print(parser_parte2)
        cantidad_bandas_parte1=len(parser_parte1)
        cantidad_bandas_parte2=cantidad_bandas-len(parser_parte1)
    else:
        cantidad_bandas_parte1=1
        cantidad_bandas_parte2=1
        parser_banprofile=parsear_banda(bandprofile,indexs)
        #print(parser_banprofile)
        parser_parte1=parser_banprofile[:1]
        parser_parte2=parser_banprofile[1:]

    #(7) the number of bands on a p-arm (8) the number of bands on a q-arm
    if invertido==False:
        #print("acaaaaaaaaa")
        cantidad_bandas_p_arm=cantidad_bandas_parte1
        bandas_p_arm=parser_parte1
        cantidad_bandas_q_arm=cantidad_bandas_parte2
        bandas_q_arm=parser_parte2
    else:
        cantidad_bandas_p_arm=cantidad_bandas_parte2
        bandas_p_arm=parser_parte2
        cantidad_bandas_q_arm=cantidad_bandas_parte1
        bandas_q_arm=parser_parte1

    #print(bandas_p_arm)
    #print(bandas_q_arm)

    #print("parserBW: "+str(bandBW))

    #11) the total number of black bands in the chromosome
    #(12) the total number of white bands in a chromosome.
    aa=[]
    #filtro para dejar un 255 o 0 en su posisicion y luego contar. los ceros son negras 255 blanco
    for i in range(0,cantidad_bandas):
        if bandBW[0]==255: #si arranca con blanco
            if i%2 == 0:
                aa.append(255)
            else:
                aa.append(0)
        if bandBW[0]==0: #si arranca con negro
            if i%2 == 0:
                aa.append(0)
            else:
                aa.append(255)


    total_black=0
    total_white=0
    for i in range(0,len(aa)):
        if aa[i]==0:
            total_black=total_black+1
        else:
           total_white=total_white+1

    #(9) the number of black bands on a p-arm, (10) the number of black bands on a q-arm
    #p-arm black - white
    parserBandBW=parsear_banda(bandBW,indexs)
    p_arm_BW=parserBandBW[:cantidad_bandas_p_arm]
    q_arm_BW=parserBandBW[cantidad_bandas_p_arm:]
    aa=[]
    p_total_black=0
    p_total_white=0
    for i in range(0,len(p_arm_BW)):
        aa.append(p_arm_BW[i][0])
    for i in range(0,len(aa)):
        if aa[i]==0:
            p_total_black=p_total_black+1
        else:
            p_total_white=p_total_white+1

    #p-arm black - white
    aa=[]
    q_total_black=0
    q_total_white=0
    for i in range(0,len(q_arm_BW)):
        aa.append(q_arm_BW[i][0])
    for i in range(0,len(aa)):
        if aa[i]==0:
            q_total_black=q_total_black+1
        else:
            q_total_white=q_total_white+1


    #print(p_total_black)
    #print(p_total_white)
    #print(q_total_black)
    #print(q_total_white)

    #5) the ratio of the largest white area to the total chromosome area
    maximo=-1
    indexi=-1
    for i in range(0,len(parserBandBW)):
        if parserBandBW[i][0]==255:
            if maximo<len(parserBandBW[i]):
                maximo=len(parserBandBW[i])
                indexi=i
    ratio_largest_white_area=len(parserBandBW[indexi])/area
    #print(ratio_largest_white_area)

    return average_max,index_avr_max,ratio_largest_white_area,cantidad_bandas,cantidad_bandas_p_arm,cantidad_bandas_q_arm,p_total_black,q_total_black,total_black,total_white


def local_band_related_features(thresh,bandprofile,img,puntos_perpendiculares,minIndexD,invertido,area):
    #cv2.namedWindow("band profile",cv2.WINDOW_NORMAL)
    #cv2.imshow("band profile",bandprofile)


    median = cv2.medianBlur(img,5)
    B=densityprofile(median,puntos_perpendiculares,thresh)
    aux=np.zeros((len(B),1),np.uint8)
    fila,col=aux.shape
    for i in range(0,fila):
            aux[i,0]=int(round(B[i][0]))
    #cv2.namedWindow("band",cv2.WINDOW_NORMAL)
    #cv2.imshow("band",aux)

    #perfil idealizado de bandas
    ##INICIA ACA --- ver si aplico esto o analisis de primera y segunda derivada para las bandW
    ##por ahora lo dejo asi ...
    fila,columna=bandprofile.shape
    #print(bandprofile.shape)
    bandBW=[]
    aa=[]
    count=0
    for i in range(0,fila-1):
        aa=bandprofile[i][0]*1+bandprofile[i+1][0]*(-1)
        #print(aa)
        if aa<0:
            bandBW.append(0)
            break
        elif aa>0:
            bandBW.append(255)
            break
        elif aa==0:
            count=count+1
    #print(count)
    for i in range(1,fila):
            aa=bandprofile[i-1][0]*1+bandprofile[i][0]*(-1)
            if aa<0:
                bandBW.append(255)
            elif aa>0:
                bandBW.append(0)
            elif aa==0:
                bandBW.append(bandBW[i-1])
    ##HASTA ACA----------------------------------------------------------------------------------


    indexs=[] #band position=el indice indica la posicion inicial (el inicio) de cada banda
    for i in range(1,len(bandBW)):
        aa=bandBW[i-1]*1+bandBW[i]*(-1)
        if aa!=0:
            indexs.append(i)

    #print(cantidad_de_bandas)
    average_max,index_avr_max,ratio_largest_white_area,\
    cantidad_bandas,cantidad_bandas_p_arm,cantidad_bandas_q_arm,\
    p_total_black,q_total_black,total_black,\
    total_white=caracteristicas1(thresh,bandprofile,bandBW,img,puntos_perpendiculares,indexs,minIndexD,invertido,area)


    aux2=np.asarray(bandBW,np.uint8) #list to array
    #cv2.namedWindow("band2",cv2.WINDOW_NORMAL)
    #cv2.imshow("band2",aux2)
    caracteristicas=[average_max,index_avr_max,ratio_largest_white_area,cantidad_bandas,cantidad_bandas_p_arm,cantidad_bandas_q_arm,p_total_black,q_total_black,total_black,total_white]
    return caracteristicas
'''
     #filtro de primera derivada
    edged=auto_canny(aux2, sigma=0.33)
    #print(edged)
    cv2.namedWindow("primera derivada",cv2.WINDOW_NORMAL)
    cv2.imshow("primera derivada",edged)
    #filtro de segunda derivada
    kernel_size = 1
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
        #segunda derivada
    laplacian = cv2.Laplacian(edged,ddepth,ksize = kernel_size,scale = scale,delta = delta)
    dst = cv2.convertScaleAbs(laplacian)
'''


def coherence_filter(img, sigma, str_sigma, blend, iter_n ):
    h, w = img.shape[:2]
    gray=img
    for i in range(0,iter_n):
        #print(i)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]

        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img*(1.0 - blend) + img1*blend)
    #print('done')
    return img


def WDD1(D):
    L=len(D)
    #print(L)
    suma1=float(0)
    suma2=float(0)
    ddd=[]
    for i in range(0,L,1):
        w_i=2*(i/L)-1
        ddd.append(w_i)
        suma1=suma1+(w_i*D[i][0])
        suma2=suma2+D[i][0]
    WDD_1=suma1/suma2
    #print(math.fsum(ddd))
    #print(ddd)
    #plt.plot(ddd)
    #plt.show()
    return  WDD_1,ddd

def WDD2(D):
    L=len(D)
    #print(L)
    suma1=float(0)
    suma2=float(0)
    ddd=[]
    for i in range(0,L):
        if i<L/2:
            w_i=-4*(i/L)-1
        if i>=((L)/2):
            w_i=-4*(i/L)+3
        ddd.append(w_i)
        suma1=suma1+(w_i*D[i][0])
        suma2=suma2+D[i][0]
    WDD_2=suma1/suma2
    #print(ddd)
    #print(math.fsum(ddd))
    #plt.plot(ddd)
    #plt.show()
    return  WDD_2,ddd

def WDD3(D):
    L=len(D)
    suma1=float(0)
    suma2=float(0)
    ddd=[]
    for i in range(0,L):
        if i< L/3:
            w_i=6*(i/L)-1
        elif i>(2*L)/3:
            w_i=6*(i/L)-5
        else:
            w_i=-6*(i/L)+3

        ddd.append(w_i)
        suma1=suma1+(w_i*D[i][0])
        suma2=suma2+D[i][0]
    WDD_3=suma1/suma2
    #print(ddd)
    #print(math.fsum(ddd))
    #plt.plot(ddd)
    #plt.show()
    #print(WDD_3)
    return  WDD_3,ddd

def WDD4(D):
    L=len(D)
    suma1=float(0)
    suma2=float(0)
    ddd=[]
    for i in range(0,L):
        if i<= (1/4)*L:
            w_i=8*(i/L)-1
        elif i> (1/4)*L and i<=(1/2)*L:
            w_i=-8*(i/L)+3
        elif i> (1/2)*L and i<=(3/4)*L:
            w_i=8*(i/L)-5
        elif i>= (3/4)*L:
            w_i=-8*(i/L)+7
        ddd.append(w_i)
        suma1=suma1+(w_i*D[i][0])
        suma2=suma2+D[i][0]
    WDD_4=suma1/suma2
    return  WDD_4,ddd


def WDD5(D):
    L=len(D)
    suma1=float(0)
    suma2=float(0)
    ddd=[]
    for i in range(0,L):
        if i<= (1/4)*L:
            w_i=4*(i/L)
        elif i>(1/4)*L and i<=(3/4)*L:
            w_i=-4*(i/L)+2
        elif i>=(3/4)*L:
            w_i=4*(i/L)-4
        ddd.append(w_i)
        suma1=suma1+(w_i*D[i][0])
        suma2=suma2+D[i][0]
    WDD_5=suma1/suma2
    return  WDD_5,ddd


def WDD6(D):
    L=len(D)
    suma1=float(0)
    suma2=float(0)
    ddd=[]
    for i in range(0,L):
        if i<= (1/6)*L:
            w_i=12*(i/L)-1
        elif i>(1/6)*L and i<=(1/3)*L:
            w_i=-12*(i/L)+3
        elif i>(1/3)*L and i<=(1/2)*L:
            w_i=12*(i/L)-5
        elif i>(1/2)*L and i<=(2/3)*L:
             w_i=-12*(i/L)+7
        elif i>(2/3)*L and i<=(5/6)*L:
             w_i=12*(i/L)-9
        elif i>(5/6)*L:
             w_i=-12*(i/L)+11
        ddd.append(w_i)
        suma1=suma1+(w_i*D[i][0])
        suma2=suma2+D[i][0]
    WDD_6=suma1/suma2
    return  WDD_6,ddd

def DWDD(ddd1,ddd2,ddd3,ddd4,ddd5,ddd6,D):
    L=len(D)
    ddi=[]
    suma2=0
    for i in range(0,L):
        suma2=suma2+D[i][0]
    # DDi(x) =| D_i(x)−D_(i−1)(x) |
    for i in range(1,L):
        ddi.append(abs(D[i][0]-D[i-1][0]))
    #DWDD1
    suma=0
    for i in range(0,len(ddi)):
        suma=suma+ddd1[i+1]*ddi[i]
    dwdd1=suma/suma2
    #print(dwdd1)
    #DWDD2
    suma=0
    for i in range(0,len(ddi)):
        suma=suma+ddd2[i+1]*ddi[i]
    dwdd2=suma/suma2
    #print(dwdd2)
    #DWDD3
    suma=0
    for i in range(0,len(ddi)):
        suma=suma+ddd3[i+1]*ddi[i]
    dwdd3=suma/suma2
    #print(dwdd3)
    #DWDD4
    suma=0
    for i in range(0,len(ddi)):
        suma=suma+ddd4[i+1]*ddi[i]
    dwdd4=suma/suma2
    #print(dwdd4)
    #DWDD5
    suma=0
    for i in range(0,len(ddi)):
        suma=suma+ddd5[i+1]*ddi[i]
    dwdd5=suma/suma2
    #print(dwdd5)
    #DWDD6
    suma=0
    for i in range(0,len(ddi)):
        suma=suma+ddd6[i+1]*ddi[i]
    dwdd6=suma/suma2
    #print(dwdd6)
    return dwdd1,dwdd2,dwdd3,dwdd4,dwdd5,dwdd6


#normaliza vector (multidimensional) de perimetro y  area
def normalizar_vector(vector):
    vector_normalizado=copy.deepcopy(vector)
    import operator
    flat_list=[item for sublist in vector for item in sublist]
    min_index, min_value = min(enumerate(flat_list), key=operator.itemgetter(1))
    max_index, max_value = max(enumerate(flat_list), key=operator.itemgetter(1))
    #print(flat_list)
    for i in range(0,len(vector)):
        for j in range(0,len(vector[i])):
            #print(vector_normalizado[i][j])
            if max_value!=min_value:
                vector_normalizado[i][j]=((vector[i][j]-min_value)/(max_value-min_value))
            else:
                vector_normalizado.append(1)
    return vector_normalizado

def extender_skeleto(skeleto,th):
    _, contours, _ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sk_puntos=np.transpose(np.nonzero(skeleto)) #[fila, columna]
    bandera=True
    count=0
    if len(sk_puntos)==1:
        skeleto[sk_puntos[0][0]+1,sk_puntos[0][1]]=255
        skeleto[sk_puntos[0][0]+2,sk_puntos[0][1]]=255
        sk_puntos=np.transpose(np.nonzero(skeleto)) #[fila, columna]
    elif len(sk_puntos)==2:
        skeleto[sk_puntos[len(sk_puntos)-1][0]+1,sk_puntos[len(sk_puntos)-1][1]]=255
        sk_puntos=np.transpose(np.nonzero(skeleto)) #[fila, columna]
    while True:#extender puntos superiores
        Ax=sk_puntos[0][1]
        Ay=sk_puntos[0][0]
        Bx=sk_puntos[len(sk_puntos)-1][1]
        By=sk_puntos[len(sk_puntos)-1][0]
        v_x=Bx-Ax
        v_y=By-Ay
        mag = math.sqrt(v_x*v_x + v_y*v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        length=-1
        C_x_1 = int(round(Ax + v_x * length))
        C_y_1 = int(round(Ay + v_y * length))
        if count==0:
            auxCx=C_x_1
            auxCy=C_y_1
        #evaluar si el punto esta dentro del contorno
        #si esta dentro del contorno lo agrego y sigo para agregar uno mas
        #si no esta dentro del contorno corto y salgo del bucle
        #False, it finds whether the point is inside or outside or on the contour (it returns +1, -1, 0 respectively).
        dist = cv2.pointPolygonTest(contours[0],(C_x_1,C_y_1),False)
        if dist==-1 or dist==0: #cortar while
            break
        elif (dist==1 and auxCx==C_x_1 and auxCy==C_y_1 and count!=0):
            break
        count=count+1
        cv2.line(skeleto,(C_x_1,C_y_1),(Ax,Ay),255,1)
        sk_puntos=np.transpose(np.nonzero(skeleto)) #[fila, columna]
        #cv2.namedWindow('imgl', cv2.WINDOW_NORMAL)
        #cv2.imshow('imgl',skeleto)
        #cv2.waitKey(0)
    #print("abajo")
    count=0
    while True:#extender puntos inferiores
        Ax=sk_puntos[len(sk_puntos)-1][1]
        Ay=sk_puntos[len(sk_puntos)-1][0]
        Bx=sk_puntos[0][1]
        By=sk_puntos[0][0]
        v_x=Bx-Ax
        v_y=By-Ay
        mag = math.sqrt(v_x*v_x + v_y*v_y)
        #print(mag)
        v_x = v_x / mag
        v_y = v_y / mag
        #print(v_x)
        #print(v_y)
        length=-1
        C_x_1 = int(round(Ax + v_x * length))
        C_y_1 = int(round(Ay + v_y * length))
        #print(C_x_1)
        #print(C_y_1)
        #print(count)
        if count==0:
            auxCx=C_x_1
            auxCy=C_y_1
        #evaluar si el punto esta dentro del contorno
        #si esta dentro del contorno lo agrego y sigo para agregar uno mas
        #si no esta dentro del contorno corto y salgo del bucle
        #False, it finds whether the point is inside or outside or on the contour (it returns +1, -1, 0 respectively).
        dist = cv2.pointPolygonTest(contours[0],(C_x_1,C_y_1),False)
        #print(dist)
        if dist==-1 or dist==0: #cortar while
            break
        elif (dist==1 and auxCx==C_x_1 and auxCy==C_y_1 and count!=0): #cortar while
            break
        count=count+1
        cv2.line(skeleto,(C_x_1,C_y_1),(Ax,Ay),255,1)
        sk_puntos=np.transpose(np.nonzero(skeleto)) #[fila, columna]
    #cv2.namedWindow('imgl--', cv2.WINDOW_NORMAL)
    #cv2.imshow('imgl--',skeleto)
    #cv2.waitKey(0)
    return skeleto

def cargar_nombres(filepath):
    archivo = open(filepath, "r")
    leer_fila= archivo.readlines()
    archivo.close()
    a=[]
    for lista in leer_fila:
        # revisamos si tiene un salto de linea al final para quitarlselo.
        if lista[-1]=="\n":
            a.append(lista[:-1].split(", ")[0])
        else:
            dato=lista.split(", ")
    archivo.close()
    return a

def count_folders(path):
    files = folders = 0
    for _, dirnames, filenames in os.walk(path):
        files += len(filenames)
        folders += len(dirnames)
    return folders




def wrapper_calcular(img):
        #agrego un pixel blanco a a todo el borde la imagen, asi el skeleto trabaja bien
        bordersize=5
        imagen=cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
        if len(imagen.shape)==3:
            gray = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
        else:
            gray=imagen
        ret, thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
        thresh=proy.eliminar_objetos_pequeños(thresh,20) #eliminar objetos pequeños
        thresh=cv2.bitwise_not(proy.floodfill(cv2.bitwise_not(thresh))) #rellenar huecos
        area,perimetro_contorno,imagen_contorno=contorno(thresh)
        return area,perimetro_contorno

def calcular_perimetro2(path,name,cantidad_de_clases):
    #calcular perimetro de todos los cromosomas de la celula en estudio
    perimetros=[]
    areas=[]
    for j in range(0,cantidad_de_clases):
        per_clase=[]
        area_clase=[]
        #print(path+"clase"+str(j+1))
        cells=len([name for name in os.listdir(path+"clase"+str(j+1)) if os.path.isfile(os.path.join(path+"clase"+str(j+1), name))])
        #print(cells)
        for i in range(0,cells):
            # Load image
            #print(path+"clase"+str(j+1)+"/clase"+str(j+1)+"_"+str(i)+"_"+name)
            img = cv2.imread(path+"clase"+str(j+1)+"/clase"+str(j+1)+"_"+str(i)+"_"+name)
            area,perimetro=wrapper_calcular(img)
            per_clase.append(perimetro)
            area_clase.append(area)
        perimetros.append(per_clase)
        areas.append(area_clase)
    #print(perimetros)
    #print(areas)

    return perimetros,areas


def obtener_caracteristicas(img,perimetro_normalizado,area):
        if len(img.shape)==3:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray=img.copy()
        filas,columnas=gray.shape
        if filas>columnas:
                bordersize=int(round(filas/2))
        else:
                bordersize=input(round(columnas/2))
        gray=cv2.copyMakeBorder(gray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=255 )

        #proyeccion_h2(gray,90)
        proyecciones=proy.calcular_proyecciones(gray)
        S,dst,dst_gray=proy.analisis2(proyecciones)
        dst1=proy.cortarImagen(S,dst,dst_gray,gray,perimetro_normalizado)#dst1->imagen enderezada
        #cv2.namedWindow('aux2', cv2.WINDOW_NORMAL)
        #cv2.imshow("aux2", dst1)
        #cv2.waitKey(0)
        ret, thresh_ = cv2.threshold(dst1,250,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        thresh_= cv2.morphologyEx(thresh_, cv2.MORPH_CLOSE,kernel)#elimino huecos
        thresh_= cv2.morphologyEx(thresh_, cv2.MORPH_OPEN, kernel)#elimino ruido del contorno del cromosoma
        #thresh_=cv2.erode(thresh_,None,None,None,1)
        #thresh_=cv2.dilate(thresh_,None,None,None,1)
        thresh_=cv2.bitwise_not(proy.floodfill(cv2.bitwise_not(thresh_))) #rellenar huecos(si quedo alguno)
        thresh_=proy.eliminar_objetos_pequeños(thresh_,20)
        #area,perimetro_contorno,imagen_contorno=contorno(thresh)-> no lo vuelvo a calcular->el perimetro y el area se conserva entre el cromosoma antes y despues de enderezar el cromosoma
        skeleto,largo_del_eje_medio=mediaAxis(dst1,cv2.bitwise_not(thresh_))
        skeleto_extendido=extender_skeleto(skeleto,cv2.bitwise_not(thresh_))
        #cv2.namedWindow('imgl', cv2.WINDOW_NORMAL)
        #cv2.imshow('imgl',thresh_)
        #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        #cv2.imshow('img',skeleto_extendido)
        #cv2.waitKey(0)
        puntos_perpendiculares,img_lineas=lineas_perpendiculares(skeleto_extendido,5,2)
        #cv2.namedWindow('imgl', cv2.WINDOW_NORMAL)
        #cv2.imshow('imgl',img_lineas)
        #lineas_perpendiculares2(skeleto)
        D=densityprofile(dst1,puntos_perpendiculares,thresh_)
        SS=shapeprofile(dst1,puntos_perpendiculares,thresh_)
        bandprofile=bandingprofile(dst1,puntos_perpendiculares,thresh_)
        CI,centromero_point,minIndexD,invertido=marcarcentromero(puntos_perpendiculares,D)
        caracteristicas=local_band_related_features(thresh_,bandprofile,dst1,puntos_perpendiculares,minIndexD,invertido,area)
        wdd_1,ddd1=WDD1(D)
        wdd_2,ddd2=WDD2(D)
        wdd_3,ddd3=WDD3(D)
        wdd_4,ddd4=WDD4(D)
        wdd_5,ddd5=WDD5(D)
        wdd_6,ddd6=WDD6(D)
        dwdd=DWDD(ddd1,ddd2,ddd3,ddd4,ddd5,ddd6,D)

        #almacenar caracteristicas en vector
        #23 caracteristicas
        #CI(length)--average_max--index_avr_max--ratio_largest_white_area--cantidad_bandas
        #cantidad_bandas_p_arm-- cantidad_bandas_q_arm-- p_total_black--q_total_black
        #total_black--total_white--wdd_1--wdd_2--wdd_3--wdd_4--wdd_5--wdd_6--
        #dwdd[0],dwdd[1],dwdd[2],dwdd[3],dwdd[4],dwdd[5]
        return  CI,caracteristicas[0],caracteristicas[1],caracteristicas[2],caracteristicas[3],caracteristicas[4],caracteristicas[5],\
        caracteristicas[6],caracteristicas[7],caracteristicas[8],caracteristicas[9],wdd_1,wdd_2,wdd_3,wdd_4,wdd_5,\
        wdd_6,dwdd[0],dwdd[1],dwdd[2],dwdd[3],dwdd[4],dwdd[5]







def wrapper_obtener_caracteristicas(path,cantidad_de_clases,perimetros_normalizados,areas,name):
    caracteristicas=[]
    for j in range(0,cantidad_de_clases):
        aux_clase=[]
        cells=len([name for name in os.listdir(path+"clase"+str(j+1)) if os.path.isfile(os.path.join(path+"clase"+str(j+1), name))])
        #print(cells)

        for i in range(0,cells):
            # Load image
            #print("------")
            print(path+"clase"+str(j+1)+"/clase"+str(j+1)+"_"+str(i)+"_"+name)
            img = cv2.imread(path+"clase"+str(j+1)+"/clase"+str(j+1)+"_"+str(i)+"_"+name)
            #print(perimetros_normalizados[j][i])
            #print(areas[j][i])
            caracteris=obtener_caracteristicas(copy.deepcopy(img),copy.deepcopy(perimetros_normalizados[j][i]),copy.deepcopy(areas[j][i]))
            aux_clase.append(caracteris)
        caracteristicas.append(aux_clase)
    return caracteristicas

def guardar_caracteristicas(caracteristicas,perimetros,areas):
    num_de_clases=len(caracteristicas)
    for i in range(0,num_de_clases):#clases
        #print("clase: "+str(i))
        myfile = open("clases/clase"+str(i+1)+".csv", 'a')#abrir y agregar al final si existe- sino lo crea y agrega
        for j in range(0,len(caracteristicas[i])):#cromosoma[i]: el vector contiene todas la caracteristicas para el cromosoma de la clase i
            c_i_j=caracteristicas[i][j]
            c_i_j=list(c_i_j)
            c_i_j.insert(0,areas[i][j])
            c_i_j.insert(0,perimetros[i][j])
            wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
            wr.writerow(c_i_j)
        myfile.close()





def prosesar_caso_de_estudio2(namee,pat):
    #/ejemploWatershed/single_clase/97002319.8.tiff/clase1
    path=pat+namee+str('/')
    #print(path)
    cantidad_de_clases=count_folders(path)
    #print(count_folders(path))
    perimetros,areas=calcular_perimetro2(path,namee,cantidad_de_clases)
    perimetros_normalizaros=normalizar_vector(perimetros) #perimetros normalizados de imagen originales
    areas_normalizadas=normalizar_vector(areas)#areas normalizadas de imagen originales
    #print(len(perimetros))
    #print(len(areas))
    caracteristicas=wrapper_obtener_caracteristicas(path,cantidad_de_clases,perimetros_normalizaros,areas,namee)

    guardar_caracteristicas(caracteristicas,perimetros,areas)

#-------------------------------------------------------------
#-------------------------------------------------------------
#--------------------------------------------------------------
if __name__ == "__main__":
    from joblib import Parallel, delayed
    import multiprocessing
    path="/home/asusn56/PycharmProjects/ejemploWatershed/"#general
    pat=path+'single_clase/'#casos
    filepath=path+"pki-3_612.lis.txt"#archivos con nombres de los casos
    nombres=cargar_nombres(filepath)
    import csv
    import concurrent.futures
    for i in range(202,len(nombres)):
    #for nombre in nombres[0:100]:
        print("---------------*****---------------")
        print(i)
        print(nombres[i])
        prosesar_caso_de_estudio2(nombres[i],pat)








