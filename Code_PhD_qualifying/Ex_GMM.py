import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image



def get_pixel2(img):
    im = Image.open(img).convert("RGB")   # for png we need .convert("RGB")
    # im = Image.open('images.jpeg')
    new_image = im.resize((100, 100))

    pix = new_image.load()
    width, height = new_image.size
    channels = 3

    pixel_values = list(new_image.getdata())
    pixel_values = np.array(pixel_values).reshape((width, height, channels))

    pix_list2 = np.zeros([width*height,3], dtype = float)
    i = 0
    for x in range(width):
        for y in range(height):
            pix = pixel_values[x,y,:]
            pix_list2[i,:] = pix/255
            i+= 1

    return pix_list2


class GMM:

    def __init__(self,X,number_of_sources,iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        
    

    """Define a function which runs for iterations, iterations"""
    def run(self):
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
           
        """ 1. Set the initial mu, covariance and pi values"""
        # create a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,len(self.X[0]))) 
        # create a nxmxm covariance matrix for each source since we have m features 
        self.cov = np.zeros((self.number_of_sources,len(self.X[0]),len(self.X[0]))) 
        # --> We create symmetric covariance matrices with ones on the digonal
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)

        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        print("self pi", self.pi)
        log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
                             # if we have converged
            
        """Plot the initial state"""    

        fig1 = plt.figure(figsize=(10,10))
        ax0 = fig1.add_subplot(111)
        ax0.scatter(self.X[:,0],self.X[:,1])
        ax0.set_title('Initial state')


        for m,c in zip(self.mu,self.cov):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        plt.show()
        
        for i in range(self.iterations):               

            """E Step"""
            r_ic = np.zeros((len(self.X),len(self.cov))) # create a rxm matrix, r the number of samples

            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                co+=self.reg_cov
                mn = multivariate_normal(mean=m,cov=co)
                component_prob = [pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(self.X) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)]
                temp = p*mn.pdf(self.X)/np.sum(component_prob,axis=0)
                r_ic[:,r] = temp

            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                mu_c = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
                self.pi.append(m_c/np.sum(r_ic)) 
 
            
            """Log likelihood"""
            likeihood = []
            for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov))):
                multi_norm = multivariate_normal(self.mu[i],self.cov[j])
                likeihood.append(k* multi_norm.pdf(self.X))

            log_likelihoods.append(np.log(np.sum(likeihood)))

        fig2 = plt.figure(figsize=(10,10))
        ax1 = fig2.add_subplot(111) 
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.iterations,1),log_likelihoods)
        plt.show()
    
    """Predict the membership of an unseen, new datapoint"""
    def predict(self,Y, plot = False):
        if plot:
            # PLot the point onto the fittet gaussians
            fig3 = plt.figure(figsize=(10,10))
            ax2 = fig3.add_subplot(111)
            ax2.scatter(self.X[:,0],self.X[:,1])
        for m,c in zip(self.mu,self.cov):
            multi_normal = multivariate_normal(mean=m,cov=c)
            if plot:
                ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
                ax2.set_title('Final state')
                for y in Y:
                    ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)
                plt.show()
        prediction = [] 
        sum_prob =  np.sum([multivariate_normal(mean=mean,cov=cov).pdf(Y) for mean,cov in zip(self.mu,self.cov)])      
        for m,c in zip(self.mu,self.cov):  
            prob_c = multivariate_normal(mean=m,cov=c).pdf(Y)/sum_prob
            prediction.append(prob_c)
        # plt.show()

        # print(Y,prediction)
        return prediction

    def plot_segment(self,img):
        im = Image.open(img).convert("RGB")
        pix = im.load()
        height, width = im.size
        print('im size', im.size)
        seg_regs = np.zeros([height, width])
        i = 0
        for x in range(height):
            for y in range(width):
                pix_vec = pix[x,y]
                pix_vec = [[a/255 for a in pix_vec]]

                prediction = self.predict(pix_vec)
                a = np.argmax(prediction) # a = 0,1,2 since num of clusters = 3
                seg_regs[x,y] = a              
                i += 1
                # print(i, prediction)

        count = (seg_regs == 0).sum()
        print('count 0', count)
        count = (seg_regs == 1).sum()
        print('count 1', count)
        count = (seg_regs == 2).sum()
        print('count 2', count)

        fig4 = plt.figure(figsize=(10,10))
        seg_regs = np.transpose(seg_regs)
        plt.imshow(seg_regs, interpolation='nearest')
        plt.show()


def main():
    # X,y = make_blobs(cluster_std = 1.5, random_state = 20, n_samples = 500, n_features=3, centers =3)

    # X =  get_pixel2('star.png')
    X =  get_pixel2('images.jpeg')
    gmm = GMM(X,4,70)
    gmm.run()
    # print('X', type(X[20][0]), X[65])
    gmm.predict([X[10]], plot = True)
    # gmm.plot_segment('star.png')
    gmm.plot_segment('images.jpeg')



if __name__ == '__main__':
    main()
