__authors__ = ['1669698', '1668784']
__group__ = '61'


from KNN import KNN
from Kmeans import KMeans, distance, get_colors
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means
import utils
import numpy as np
import time
import matplotlib.pyplot as plt

def Execute_kmeans_images(image_set, k_max, init = {}):
    """
    Function to execute KMeans on a set of images
    :param image_set: list of images
    :param k_max: maximum number of clusters
    :return: list of KMeans objects for each image
    """
    images_km = np.empty((0,), dtype=object)

    for image in image_set:
        image_to_km = KMeans(image, K=1 ,options=init)
        image_to_km.find_bestK(k_max)
        image_to_km.fit()
        images_km = np.append(images_km, image_to_km)
    
    return images_km

def Execute_classifications(image_set,image_train,class_train,k_max = 10, k_knn=4, init = {}):
    """
    Function to execute KMeans on a set of images
    :param image_set: list of images
    :param k_max: maximum number of clusters
    :return: list of KMeans objects for each image
    :return: list of colors for each image
    :return: list of shapes for each image
    """
    trained_knn = KNN(utils.rgb2gray(image_train), class_train)
    images_km = np.empty((0,), dtype=object)
    images_colors = []
    images_shapes = trained_knn.predict(utils.rgb2gray(image_set), k_knn)
    for image in image_set:
        image_to_km = KMeans(image,K=1, options=init)
        image_to_km.find_bestK(k_max)
        image_to_km.fit()
        images_km = np.append(images_km, image_to_km)
        
        colors = get_colors(image_to_km.centroids)
        #print(len(colors))
        images_colors.append(colors)
        
        #print(colors[1])
        #predicted_shape = trained_knn.predict(image, 10)
        #images_shapes = np.append(utils.rgb2gray(images_shapes), predicted_shape)
    
    return images_km, images_colors, images_shapes

def Retrieval_by_color(images, colors, query, or_cond = True):
    """
    Function to retrieve images based on color
    :param images: list of images
    :param colors: list of colors
    :param query: query image
    :return: list of retrieved images
    """
    #images_founded = np.empty((0,) + images[0].shape)
    images_founded = []
    visited_images = np.empty((0,), dtype=int)
    for color_query in query:
        for i in range(len(images)):
            if or_cond and color_query in colors[i] and i not in visited_images:
                images_founded.append(images[i])
                visited_images = np.append(visited_images, i)
            elif all(color_query in colors[i] for color_query in query) and i not in visited_images:
                images_founded.append(images[i])
                visited_images = np.append(visited_images, i)
    return images_founded

def Retrieval_by_shape(images, shape, query):
    """
    Function to retrieve images based on color
    :param images: list of images
    :param shape: list of shape
    :param query: query image
    :return: list of retrieved images
    """
    #images_founded = np.empty((0,) + images[0].shape)
    images_founded = []
    visited_images = np.empty((0,), dtype=int)
    for color_query in query:
        for i in range(len(images)):
            if color_query in shape[i] and i not in visited_images:
                #images_founded = np.append(images_with_color, np.array([np.copy(images[i])]), axis=0)
                images_founded.append(images[i])
                visited_images = np.append(visited_images, i)
    return images_founded  
            
def Retrieval_combined(images, color, shape, color_querys, shape_querys):
    """
    Function to retrieve images based on color
    :param images: list of images
    :param shape: list of shape
    :param query: query image
    :return: list of retrieved images
    """
    #images_founded = np.empty((0,) + images[0].shape)
    images_founded = []
    visited_images = np.empty((0,), dtype=int)
    for shape_query, color_query in zip(shape_querys, color_querys):
        for i in range(len(images)):
            if shape_query in shape[i] and color_query in color[i] and i not in visited_images:
                #images_founded = np.append(images_with_color, np.array([np.copy(images[i])]), axis=0)
                images_founded.append(images[i])
                visited_images = np.append(visited_images, i)
    return images_founded

def Kmean_statistics(km, Kmax):
    """
    Analitza el rendiment de K-means per diferents valors de K
    """
    wcd_values = []     #distàncies intra-cluster
    bcd_values = []     #distàncies inter-cluster
    fisher_values = []  #relació de Fisher
    iteracions = []     #nombre d'iteracions
    temps_exec = []     #temps d'execució
    
    for k in range(2, Kmax+1):
        #configura nova K i reinicia l'estat
        km.K = k
        km._init_centroids()
        
        start_time = time.time()
        km.fit()
        elapsed = time.time() - start_time
        
        #recull mètriques
        wcd_values.append(km.withinClassDistance())
        bcd_values.append(km.between_class_distance())
        fisher_values.append(km.fisher_ratio())
        iteracions.append(km.num_iter)
        temps_exec.append(elapsed)
    
    #visualització millorada
    plt.figure(figsize=(20,8))
    
    #gràfic 1: WCD
    plt.subplot(2,3,1)
    plt.plot(range(2,Kmax+1), wcd_values, 'bo-')
    plt.title("Distàncies Intra-classe")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Distància")
    
    #gràfic 2: BCD
    plt.subplot(2,3,2)
    plt.plot(range(2,Kmax+1), bcd_values, 'go-')
    plt.title("Distàncies Inter-classe")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Distància")
    
    #gràfic 2: Relació de Fisher
    plt.subplot(2,3,3)
    plt.plot(range(2,Kmax+1), fisher_values, 'mo-')
    plt.title("Relació de Fisher (WCD/BCD)")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("F-ratio")
    
    #gràfic 3: Iteracions
    plt.subplot(2,3,4)
    plt.bar(range(2,Kmax+1), iteracions)
    plt.title("Iteracions per Convergència")
    plt.xlabel("K")
    
    #gràfic 4: Temps
    plt.subplot(2,3,5)
    plt.plot(range(2,Kmax+1), temps_exec, 'r^-')
    plt.title("Temps d'Execució (s)")
    plt.xlabel("K")
    
    plt.tight_layout()
    plt.show()
    
def Get_shape_accuracy(prediccions, ground_truth):
    """
    Calcula el percentatge d'encert en la classificació de formes
    """
    #convertim a arrays numpy per facilitar les operacions
    preds = np.array(prediccions)
    gt = np.array(ground_truth)
    
    #comprovem que tenen la mateixa longitud
    if len(preds) != len(gt):
        raise ValueError("Les llistes de prediccions i Ground-Truth han de tenir la mateixa longitud")
    
    #calculem el nombre de coincidències
    coincidencies = np.sum(preds == gt)
    
    #calculem el percentatge
    percentatge = (coincidencies / len(gt)) * 100
    
    return percentatge

def Get_color_accuracy(prediccions, ground_truth):
    """
    Calcula la similitud entre colors predits i reals usant l'índex de Jaccard
    """
    jaccard_total = 0.0
    
    for colors_pred, colors_real in zip(prediccions, ground_truth):
        #convertim a conjunts per facilitar les operacions
        set_pred = set(colors_pred)
        set_real = set(colors_real)
        
        #calculem intersecció i unió
        interseccio = len(set_pred & set_real)
        unio = len(set_pred | set_real)
        
        #evitem la divisió per zero (cas de dos conjunts buits)
        if unio == 0:
            jaccard = 1.0  #considerem coincidència perfecta si ambdós són buits
        else:
            jaccard = interseccio / unio
        
        jaccard_total += jaccard
    
    #mitjana de Jaccard en percentatge
    return (jaccard_total / len(prediccions)) * 100

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    avaliable_colors = ['Black', 'White', 'Red', 'Green', 'Blue', 'Yellow', 'Grey', 'Orange', 'Brown', 'Pink', 'Purple']
    avaliable_options = [{'km_init':'first'}, {'km_init': 'random'}, {'km_init': 'grey'}, {'km_init': 'color_prob'}, {'km_init': 'color_dispersed'}]
    
    #for available_option in avaliable_options:
    #    ###EXECUCIÓ ESTADÍSTICA KMEANS###
    #    from Kmeans import KMeans
    #    km = KMeans(test_imgs[4], K=2, options=available_option)
    #
    #    km.options['fitting'] = 'WCD'
    #    km.find_bestK(10)
    #    print(f"Millor K {km.options['fitting']}: {km.K}")
    #    
    #    km.options['fitting'] = 'BCD'
    #    km.find_bestK(10)
    #    print(f"Millor K {km.options['fitting']}: {km.K}")
    #    
    #    km.options['fitting'] = 'FISHER'
    #    km.find_bestK(10)
    #    print(f"Millor K {km.options['fitting']}: {km.K}")
    #
    #    Kmean_statistics(km, Kmax=10)
    #    
    #    H, W, C = test_imgs[4].shape
    #    visualize_k_means(km, (H, W, C))
    #    ########################
    
    ###EXECUCIÓ ESTADÍSTICA KMEANS SENSE KM_INIT###
    #from Kmeans import KMeans
    #km = KMeans(test_imgs, K=2)
    #
    #km.options['fitting'] = 'WCD'
    #km.find_bestK(10)
    #print(f"Millor K {km.options['fitting']}: {km.K}")
    #
    #km.options['fitting'] = 'BCD'
    #km.find_bestK(10)
    #print(f"Millor K {km.options['fitting']}: {km.K}")
    #
    #km.options['fitting'] = 'FISHER'
    #km.find_bestK(10)
    #print(f"Millor K {km.options['fitting']}: {km.K}")
    #
    #Kmean_statistics(km, Kmax=10)
    ########################
    
    ###EXECUCIÓ SHAPE ACCURACY###
    
    ##execució del KNN per obtenir prediccions
    #trained_knn = KNN(utils.rgb2gray(test_imgs), test_class_labels)
    #predicted_shapes = trained_knn.predict(utils.rgb2gray(test_imgs), k=4) ##!!! EXCEPTUANT k=1 i k=2, <k=4> SEMBLA DONAR EL MILLOR RESULTAT => k=4 !!!##
    #
    ##càlcul de l'encert
    #accuracy = Get_shape_accuracy(predicted_shapes, test_class_labels)
    #total_samples = len(test_class_labels)
    #error_count = np.sum(np.array(predicted_shapes) != np.array(test_class_labels))
    #
    #print(f"Precisió en la classificació de formes: {accuracy:.2f}%")
    #print(f"Errors: {error_count}/{total_samples}")
    #
    ##visualització d'exemples incorrectes amb més informació
    #incorrectes = np.where(np.array(predicted_shapes) != np.array(test_class_labels))[0]
    #if len(incorrectes) > 0:
    #    #preparem la informació ampliada
    #    error_info = [
    #        f"Predicció: {predicted_shapes[i]}\nReal: {test_class_labels[i]}" 
    #        for i in incorrectes[:8]
    #    ]
    #    
    #    #creem el títol amb les estadístiques
    #    plot_title = (
    #        f"Classificacions incorrectes\n"
    #        f"Precisió: {accuracy:.2f}% | Errors: {error_count}/{total_samples}"
    #    )
    #    
    #    visualize_retrieval(
    #        test_imgs[incorrectes[:8]], 
    #        topN=8,
    #        info=error_info,
    #        ok=[False]*8,
    #        title=plot_title,
    #        query=None
    #    )
    
    #############################
    
    ###EXECUCIÓ COLOR ACCURACY###
    
    #options = {'km_init': 'grey', 'fitting': 'WCD'}
    #images_km, predicted_colors, images_shapes = Execute_classifications(test_imgs, test_imgs, test_class_labels, init = options)
    #
    ##processar totes les imatges de test
    #for img in test_imgs:
    #    km = KMeans(img, K=5)
    #    km.fit()
    #    predicted_colors.append(get_colors(km.centroids))
    #
    ##calcular precisió total
    #
    #print(f"Analitzant {len(test_imgs)} imatges\n")
    #color_accuracy = Get_color_accuracy(predicted_colors, test_color_labels)
    #print(f"\nSimilitud mitjana total en colors: {color_accuracy:.2f}%")
    #
    #for i in range(5):
    #    #calcular Jaccard individual
    #    set_pred = set(predicted_colors[i])
    #    set_real = set(test_color_labels[i])
    #    interseccio = len(set_pred & set_real)
    #    unio = len(set_pred | set_real)
    #    similitud = (interseccio / unio) * 100 if unio > 0 else 100
    #    
    #    #visualització
    #    print(f"Imatge {i}:")
    #    print(f"  Predicció: {', '.join(predicted_colors[i])}")
    #    print(f"  Real:      {', '.join(test_color_labels[i])}")
    #    print(f"  Coincidències: {interseccio} de {unio} colors")
    #    print(f"  Similitud: {similitud:.2f}%")
    #    print("-"*50)
    
    #############################
    


    #for type in ["WCD","BCD","FISHER"]:
    #   for available_option in avaliable_options:
    #       #available_option['fitting'] = type
    #       km_images, colors, shapes = Execute_classifications(train_imgs,train_imgs,train_class_labels,k_max = 10, k_knn=4, init = available_option)
    #       color_accuracy = Get_color_accuracy(colors, color_labels)
    #       print(f"Opció: {available_option}")
    #       print(f"Similitud mitjana total en colors: {color_accuracy:.2f}%\n")



    ############################### TESTS QUALITATIUS ###############################


    #color_images_retrival = Retrieval_by_color(imgs, color_labels, ["Red","Green"], or_cond = False)
    #visualize_retrieval(color_images_retrival, 12)
    
    #label_images_retrival = Retrieval_by_shape(test_imgs, test_class_labels, ["Socks", "Jeans"])
    #visualize_retrieval(label_images_retrival, 12)
    
    #combo_images_retrival = Retrieval_combined(test_imgs, test_color_labels, test_class_labels, ["Green", "Blue"], ["Socks", "Jeans"])
    #visualize_retrieval(combo_images_retrival, 12)
    
    #combo_images_retrival = Retrieval_combined(test_imgs, test_color_labels, test_class_labels, ["Green"], ["Socks"])
    #visualize_retrieval(combo_images_retrival, 12)
    
    # Inicialitza KMeans per a les imatges de test
    #my_image_to_km, my_image_color, my_image_shape = Execute_classifications(imgs,train_imgs,train_class_labels, k_max = 10, k_knn=4, init = {})
    #combo_images_retrival = Retrieval_by_color(imgs, my_image_color, ["Red","Green"], or_cond = False)
    #visualize_retrieval(combo_images_retrival, 12)
    #label_images_retrival = Retrieval_by_shape(imgs, my_image_shape, ["Socks", "Jeans"])
    #visualize_retrieval(label_images_retrival, 12)
    #combo_images_retrival = Retrieval_combined(imgs, my_image_color, my_image_shape, ["Green", "Grey"], ["Socks", "Heels"])
    #visualize_retrieval(combo_images_retrival, 12)
    
    #combo_images_retrival = Retrieval_by_color(imgs, my_image_color, [avaliable_colors[3],avaliable_colors[1]])
    #visualize_retrieval(combo_images_retrival, 12)
    
    
    ################Calculem el nº d'imatges per cada color################
    #dict_n_colors = {}
    #for color in avaliable_colors:
    #    combo_images_retrival = Retrieval_by_color(train_imgs, train_color_labels, [color])
    #    dict_n_colors[color] = len(combo_images_retrival)
    #print(dict(sorted(dict_n_colors.items(), key=lambda item: item[1], reverse=True))) 
    #"""
    #{
    #    'White': 2328,
    #    'Grey': 1242,
    #    'Black': 1148,
    #    'Orange': 1030,
    #    'Blue': 555,
    #    'Brown': 426,
    #    'Red': 361,
    #    'Pink': 331,
    #    'Yellow': 220,
    #    'Green': 171,
    #    'Purple': 128
    #}
    #"""