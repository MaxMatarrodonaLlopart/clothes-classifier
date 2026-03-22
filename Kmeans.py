__authors__ = ['1669698', '1668784']
__group__ = '61'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    #!#  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        if not np.issubdtype(X.dtype, np.floating):
            self.X = X.astype(float)
        else:
            self.X = X
            
        shape = self.X.shape
        if len(shape) > 2:
            self.X = self.X.reshape(-1, 3)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        #!#  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        
        
        ############  INIT FIRST  ############
        if self.options['km_init'].lower() == 'first': #si el mètode d'inicialització és 'first'
            centroids = [] #inicialitzem una llista buida de centroides
            seen = set() #inicialitzem un set buit per evitar duplicats i guardar els centroides ja vistos
            for point in self.X: #iterem sobre cada punt
                tuple_point = tuple(np.round(point, decimals=4)) #arrodonim el punt a 4 decimals per evitar errors i el convertim a tuple
                if tuple_point not in seen: #si el punt no ha estat vist abans
                    seen.add(tuple_point) #afegim el punt al set de vistos
                    centroids.append(point) #afegim el punt a la llista de centroides
                    if len(centroids) == self.K: #si ja tenim K centroides, sortim del bucle
                        break
            if len(centroids) < self.K: #si no hem trobat K centroides únics
                remaining = self.K - len(centroids) #calculem quants ens falten
                centroids.extend(self.X[:remaining]) #afegim els primers remaining punts de X a la llista de centroides
            self.centroids = np.array(centroids) #convertim la llista de centroides a un array numpy
        ############  INIT FIRST  ############
        
        
        
        ############  INIT RANDOM  ############
        elif self.options['km_init'].lower() == 'random': #Si el mètode d'inicialització és 'random'
            indices = np.random.permutation(len(self.X)) #generem una permutació aleatòria dels índexs de X
            centroids = [] #inicialitzem una llista buida de centroides
            seen = set() #inicialitzem un set buit per evitar duplicats i guardar els centroides ja vistos
            for idx in indices: #iterem sobre els índexs aleatoris
                point = self.X[idx] #agafem el punt corresponent a l'índex
                tuple_point = tuple(np.round(point, decimals=4)) #arrodonim el punt a 4 decimals per evitar errors i el convertim a tuple
                if tuple_point not in seen: #si el punt no ha estat vist abans
                    seen.add(tuple_point) #afegim el punt al set de vistos
                    centroids.append(point) #afegim el punt a la llista de centroides
                    if len(centroids) == self.K: #si ja tenim K centroides, sortim del bucle
                        break
            if len(centroids) < self.K: #si no hem trobat K centroides únics
                remaining = self.K - len(centroids) #calculem quants ens falten
                centroids.extend(self.X[:remaining]) #afegim els primers remaining punts de X a la llista de centroides
            self.centroids = np.array(centroids) #convertim la llista de centroides a un array numpy
        ############  INIT RANDOM  ############
            
            
            
        ############  INIT GREY  ############
        elif self.options['km_init'].lower() == 'grey': #si el mètode d'inicialització és 'custom'
            min_vals = np.min(self.X, axis=0) #calculem el mínim de cada dimensió de X
            max_vals = np.max(self.X, axis=0) #calculem el màxim de cada dimensió de X
            if self.K > 1: #si K és més gran que 1
                step = (max_vals - min_vals) / (self.K - 1) #calculem l'espaiat entre centroides
            else: #si K és 1, no cal espaiar els centroides
                step = 0
            centroids = [min_vals + i * step for i in range(self.K)] #creem una llista de centroides espaiats uniformement entre el mínim i el màxim de X
            self.centroids = np.array(centroids) #convertim la llista de centroides a un array numpy
        ############  INIT GREY  ############ 
            
            
            
        ############  INIT COLOR PROB  ############
        elif self.options['km_init'].lower() == 'color_prob':
            color_centroids = [
                        [255.0, 255.0, 255.0],  #'White': 2328, => rgb(255.0, 255.0, 255.0)
                        [128.0, 128.0, 128.0],  #'Grey': 1242, => rgb(128.0, 128.0, 128.0)
                        [0.0, 0.0, 0.0],        #'Black': 1148, => rgb(0.0, 0.0, 0.0)
                        [255.0, 165.0, 0.0],    #'Orange': 1030, => rgb(255.0, 165.0, 0.0)
                        [0.0, 0.0, 255.0],      #'Blue': 555, => rgb(0.0, 0.0, 255.0)
                        [165.0, 42.0, 42.0],    #'Brown': 426, => rgb(165.0, 42.0, 42.0)
                        [255.0, 0.0, 0.0],      #'Red': 361, => rgb(255.0, 0.0, 0.0)
                        [255.0, 192.0, 203.0],  #'Pink': 331, => rgb(255.0, 192.0, 203.0)
                        [255.0, 255.0, 0.0],    #'Yellow': 220, => rgb(255.0, 255.0, 0.0)
                        [0.0, 255.0, 0.0],      #'Green': 171, => rgb(0.0, 255.0, 0.0)
                        [128.0, 0.0, 128.0]]    #'Purple': 128 => rgb(128.0, 0.0, 128.0)
            self.centroids = np.array(color_centroids[0:self.K]) #assignem els primers K centroides de la llista de colors (els més probables)
        ############  INIT COLOR PROB  ############
            
            
            
        ############  INIT COLOR DISPERSED  ############
        elif self.options['km_init'].lower() == 'color_dispersed': ## Variació de color_prob, sembla no funcionar tant bé ##
            color_centroids = [
                        [255.0, 255.0, 255.0],  #'White': 2328, => rgb(255.0, 255.0, 255.0)
                        [0.0, 0.0, 0.0],        #'Black': 1148, => rgb(0.0, 0.0, 0.0)
                        [128.0, 128.0, 128.0],  #'Grey': 1242, => rgb(128.0, 128.0, 128.0)
                        [0.0, 0.0, 255.0],      #'Blue': 555, => rgb(0.0, 0.0, 255.0)
                        [255.0, 0.0, 0.0],      #'Red': 361, => rgb(255.0, 0.0, 0.0)
                        [0.0, 255.0, 0.0],      #'Green': 171, => rgb(0.0, 255.0, 0.0)
                        [255.0, 255.0, 0.0],    #'Yellow': 220, => rgb(255.0, 255.0, 0.0)
                        [128.0, 0.0, 128.0],    #'Purple': 128 => rgb(128.0, 0.0, 128.0)
                        [255.0, 192.0, 203.0],  #'Pink': 331, => rgb(255.0, 192.0, 203.0)
                        [255.0, 165.0, 0.0],    #'Orange': 1030, => rgb(255.0, 165.0, 0.0)
                        [165.0, 42.0, 42.0]]    #'Brown': 426, => rgb(165.0, 42.0, 42.0)
            self.centroids = np.array(color_centroids[0:self.K]) #assignem els primers K centroides de la llista de colors (els més probables)
        ############  INIT COLOR DISPERSED  ############
            
            
            
        ############  COPY TO OLD  ############
        self.old_centroids = np.copy(self.centroids) #inicialitzem old_centroids com una copia dels centroides actuals
        ############  COPY TO OLD  ############
        
    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        
        dist_matrix = distance(self.X, self.centroids) #calculem la distància entre cada punt de X i cada centroide
        self.labels = np.argmin(dist_matrix, axis=1) #troba l'etiqueta del centroide més proper per a cada punt de X

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #!######################################################
        #!#  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #!#  AND CHANGE FOR YOUR OWN CODE
        #!######################################################
        
        self.old_centroids = np.copy(self.centroids)
        
        
        #for i in range(self.K):
        #    if np.any(self.labels == i):
        #        self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)
        #    else:
        #        self.centroids[i] = np.random.rand(self.X.shape[1])
            
        for i in range(self.K):
            self.old_centroids[i] = self.centroids[i]
            if np.any(self.labels == i):  #comprovem si hi ha punts assignats al centroide i si és així, calculem la mitjana
                self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)
            else:
                #reinicialitzem el centroide amb un punt aleatori de X
                self.centroids[i] = self.X[np.random.randint(0, self.X.shape[0])]
        pass

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        
        for i in range(self.K):
            if not np.array_equal(self.old_centroids[i], self.centroids[i]):
                return False
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        
        self._init_centroids() #inicialitzem els centroides
        self.num_iter = 0 #inicialitzem el nombre d'iteracions a 0

        while True:
            self.get_labels() #assignem cada punt al centroide més proper
            self.get_centroids() #calculem les noves coordenades dels centroides
            if self.converges() or self.num_iter >= self.options['max_iter']: #si els centroides no canvien o hem arribat al màxim d'iteracions
                break

            self.num_iter += 1 #incrementem el nombre d'iteracions

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #!######################################################
        #!#  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #!#  AND CHANGE FOR YOUR OWN CODE
        #!######################################################

        return np.mean(np.sum((self.X- self.centroids[self.labels]) ** 2, axis=1))
    
    def between_class_distance(self):
        """
        Calcula la distància inter-classe (Between-Class Distance)
        """
        centroids = self.centroids
        global_mean = np.mean(self.X, axis=0)
        total = 0.0
        
        for k in range(self.K):
            cluster_points = self.X[self.labels == k]
            nk = len(cluster_points)
            total += nk * np.linalg.norm(centroids[k] - global_mean)**2
            
        return total / len(self.X)

    def fisher_ratio(self):
        """
        Calcula la relació de Fisher (F-ratio): WCD / BCD
        """
        wcd = self.withinClassDistance()
        bcd = self.between_class_distance()
        return wcd / bcd if bcd != 0 else 0
    
    def find_bestK(self, max_K):
        """
        Determina la millor K segons l'heurística especificada a self.options['bestk_method']
        Opcions: 'WCD' (Within-Class), 'BCD' (Between-Class), 'FISHER' (Ratio)
        """
        fitting = self.options['fitting'].lower() 

        if fitting == 'wcd':
            minimum = 15
            """
             sets the best k analysing the results up to 'max_K' clusters
            """
            #!######################################################
            #!#  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
            #!#  AND CHANGE FOR YOUR OWN CODE
            #!######################################################
            old_WCDk = np.inf
            self.K = 1
            while self.K < max_K + 1:
                self.fit()
                WCDk = self.withinClassDistance()
                DECk = 100 * WCDk / old_WCDk
                old_WCDk = WCDk
                if 100 - DECk < minimum: #llindar a canviar
                    self.K -= 1
                    return 0
                self.K += 1
            pass

        elif fitting == 'bcd':
            minimum = 0.1
        
            old_BCD = -np.inf
            self.K = 2
            while self.K < max_K + 1:
                self.fit()
                BCD = self.between_class_distance()
                DECk = 100 * old_BCD / BCD
                old_BCD = BCD
                if 100 - DECk < minimum: #llindar a canviar
                    self.K -= 1
                    return 0
                self.K += 1
            pass

        elif fitting == 'fisher':
            minimum = 15
        
            old_fisher = -np.inf
            self.K = 2
            while self.K < max_K + 1:
                self.fit()
                fisher = self.fisher_ratio()
                DECk = 100 * fisher / old_fisher
                old_fisher = fisher
                if 100 - DECk < minimum: #llindar a canviar
                    self.K -= 1
                    return 0
                self.K += 1
            pass

        else:
            raise ValueError(f"Mètode desconegut: {fitting}. Opcions vàlides: WCD, BCD, FISHER")

    ###FIND BEST K ORIGINAL###
    
    def find_bestK(self, max_K):
        minimum = 20
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #!######################################################
        #!#  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #!#  AND CHANGE FOR YOUR OWN CODE
        #!######################################################
        old_WCDk = np.inf
        self.K = 1
        while self.K < max_K + 1:
            self.fit()
            WCDk = self.withinClassDistance()
            DECk = 100 * WCDk / old_WCDk
            old_WCDk = WCDk
            if 100 - DECk < minimum: #llindar a canviar
                self.K -= 1
                return 0
            self.K += 1
        pass
    #
    ###########################

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    
    # np.linalg.norm calcula la distància eucladiana entre tots els elements de 2 vectors
    # X[:, np.newaxis] afegeix una nova dimensió a X, per tal de poder restar els dos arrays
    
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2) 


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    color_probs = utils.get_color_prob(centroids) #obtenim les probabilitats de pertànyer a cada color
    color_indices = np.argmax(color_probs, axis=1) #obtenim el color més probable per a cada centroide
    color_labels = [] #inicialitzem una llista buida de labels de colors
    
    for idx in color_indices: #iterem sobre els índexs dels colors
        color_name = utils.colors[idx] #obtenim el nom del color corresponent a l'índex
        color_labels.append(color_name) #afegim el nom del color a la llista de labels de colors

    return color_labels #retornem la llista de labels de colors