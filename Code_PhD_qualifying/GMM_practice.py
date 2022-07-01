import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import multivariate_normal


# def extract_vector


from PIL import Image

def get_pixel():
    im = Image.open('tree.jpg')
    im = Image.open('my.png')
    pix = im.load()

    # print(im.size)  # Get the width and hight of the image for iterating over
    # print(pix[10,10])  # Get the RGBA Value of the a pixel of an image

    # print(pix)
    width, height = im.size
    channels = 3

    pixel_values = list(im.getdata())
    pixel_values = np.array(pixel_values).reshape((width, height, channels))


    print(pixel_values.shape)

    pix1 = pixel_values[0,1,:]
    # print(pix1)
    # print(pixel_values[0].shape)

    pix_list = []
    pix_list2 = np.zeros([width*height,3], dtype = float)
    i = 0
    for x in range(width):
        for y in range(height):
            pix = pixel_values[x,y,:]
            pix_list.append(pix)
            pix_list2[i,:] = pix/255
            i+= 1
    print(len(pix_list))
    print(np.array(pix_list).shape)
    # return (np.array(pix_list)/255) #pix_list
    return pix_list2[:500,:]


def get_pixel2(img):
    im = Image.open(img).convert("RGB")   # for png we need .convert("RGB")
    # im = Image.open('images.jpeg')
    # im = Image.open('rgb.png').convert("RGB")
    new_image = im.resize((100, 100))
    # new_image.save('image_100.jpg')

    # im = Image.open('my.png')
    pix = new_image.load()
    # print(im.size)  # Get the width and hight of the image for iterating over
    # print(pix[10,10])  # Get the RGBA Value of the a pixel of an image

    # print(pix)
    width, height = new_image.size
    channels = 3

    pixel_values = list(new_image.getdata())
    pixel_values = np.array(pixel_values).reshape((width, height, channels))


    print(pixel_values.shape)

    pix1 = pixel_values[0,1,:]
    # print(pix1)
    # print(pixel_values[0].shape)

    pix_list = []
    pix_list2 = np.zeros([width*height,3], dtype = float)
    i = 0
    for x in range(width):
        for y in range(height):
            pix = pixel_values[x,y,:]
            pix_list.append(pix)
            pix_list2[i,:] = pix/255
            i+= 1
    print(len(pix_list))
    print(np.array(pix_list).shape)
    # return (np.array(pix_list)/255) #pix_list
    return pix_list2


class GMM:

    def __init__(self,X,number_of_sources,iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        
    

    """Define a function which runs for iterations, iterations"""
    def run(self):
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T
           
        print("RAT", self.X[:,0].shape)
        """ 1. Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,len(self.X[0]))) # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        print('MU', self.mu)
        self.cov = np.zeros((self.number_of_sources,len(self.X[0]),len(self.X[0]))) # We need a nxmxm covariance matrix for each source since we have m features --> We create symmetric covariance matrices with ones on the digonal
        # print('COV', self.cov)
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)
        print('COV', self.cov)

        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        print("self pi", self.pi)
        log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
                             # if we have converged
            
        """Plot the initial state"""    
        # fig = plt.figure(figsize=(10,10))
        # ax0 = fig.add_subplot(311)
        # ax0.scatter(self.X[:,0],self.X[:,1])
        # ax0 = fig.add_subplot(312)
        # ax0.scatter(self.X[:,1],self.X[:,2])
        # ax0 = fig.add_subplot(313)
        # ax0.scatter(self.X[:,0],self.X[:,2])

        # ax0.set_title('Initial state')
        # plt.show()

        fig1 = plt.figure(figsize=(10,10))
        ax0 = fig1.add_subplot(111)
        ax0.scatter(self.X[:,0],self.X[:,1])
        ax0.set_title('Initial state')
        # plt.show()

        # fig2 = plt.figure(figsize=(10,10))
        # ax2 = fig2.add_subplot(111)
        # ax2.scatter(self.X[:,0],self.X[:,2])
        # ax2.set_title('Initial state')
        # plt.show()

        for m,c in zip(self.mu,self.cov):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m,cov=c)
            # ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
            ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        plt.show()
        
        for i in range(self.iterations):               

            """E Step"""
            r_ic = np.zeros((len(self.X),len(self.cov)))

            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                co+=self.reg_cov
                # print("CO", co, m)
                mn = multivariate_normal(mean=m,cov=co)
                # print("MNX", mn.pdf(self.X))
                temp = p*mn.pdf(self.X)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(self.X) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)
                # print([multivariate_normal(mean=mu_c,cov=cov_c).pdf(self.X) for mu_c,cov_c in zip(self.mu,self.cov+self.reg_cov)])
                # print("TEMP", temp)
                r_ic[:,r] = temp
            """
            The above calculation of r_ic is not that obvious why I want to quickly derive what we have done above.
            First of all the nominator:
            We calculate for each source c which is defined by m,co and p for every instance x_i, the multivariate_normal.pdf() value.
            For each loop this gives us a 100x1 matrix (This value divided by the denominator is then assigned to r_ic[:,r] which is in 
            the end a 100x3 matrix).
            Second the denominator:
            What we do here is, we calculate the multivariate_normal.pdf() for every instance x_i for every source c which is defined by
            pi_c, mu_c, and cov_c and write this into a list. This gives us a 3x100 matrix where we have 100 entrances per source c.
            Now the formula wants us to add up the pdf() values given by the 3 sources for each x_i. Hence we sum up this list over axis=0.
            This gives us then a list with 100 entries.
            What we have now is FOR EACH LOOP a list with 100 entries in the nominator and a list with 100 entries in the denominator
            where each element is the pdf per class c for each instance x_i (nominator) respectively the summed pdf's of classes c for each 
            instance x_i. Consequently we can now divide the nominator by the denominator and have as result a list with 100 elements which we
            can then assign to r_ic[:,r] --> One row r per source c. In the end after we have done this for all three sources (three loops)
            and run from r==0 to r==2 we get a matrix with dimensionallity 100x3 which is exactly what we want.
            If we check the entries of r_ic we see that there mostly one element which is much larger than the other two. This is because
            every instance x_i is much closer to one of the three gaussians (that is, much more likely to come from this gaussian) than
            it is to the other two. That is practically speaing, r_ic gives us the fraction of the probability that x_i belongs to class
            c over the probability that x_i belonges to any of the classes c (Probability that x_i occurs given the 3 Gaussians).
            """

            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                # print("********", (r_ic[:,c].reshape(len(self.X),1).shape))
                mu_c = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
                # print('MU_C', mu_c)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
                self.pi.append(m_c/np.sum(r_ic)) # Here np.sum(r_ic) gives as result the number of instances. This is logical since we know 
                                                # that the columns of each row of r_ic adds up to 1. Since we add up all elements, we sum up all
                                                # columns per row which gives 1 and then all rows which gives then the number of instances (rows) 
                                                # in X --> Since pi_new contains the fractions of datapoints, assigned to the sources c,
                                                # The elements in pi_new must add up to 1

            
            
            """Log likelihood"""
            log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(self.X) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))

            

            """
            This process of E step followed by a M step is now iterated a number of n times. In the second step for instance,
            we use the calculated pi_new, mu_new and cov_new to calculate the new r_ic which are then used in the second M step
            to calculat the mu_new2 and cov_new2 and so on....
            """

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
                # ax2.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                
                ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
                ax2.set_title('Final state')
                for y in Y:
                    ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)
                plt.show()
        prediction = []        
        for m,c in zip(self.mu,self.cov):  
            #print(c)
            prediction.append(multivariate_normal(mean=m,cov=c).pdf(Y)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(Y) for mean,cov in zip(self.mu,self.cov)]))
        # plt.show()

        # print(Y,prediction)
        return prediction

    def plot_segment(self,img):
        im = Image.open(img).convert("RGB")
        pix = im.load()
        height, width = im.size
        seg_regs = np.zeros([height, width])
        i = 0
        for x in range(height):
            for y in range(width):
                pix_vec = pix[x,y]
                pix_vec = [[a/255 for a in pix_vec]]
                
                # print(i, 'pix_vec', pix_vec)

                prediction = self.predict(pix_vec)
                # prediction = []
                # for m,c in zip(self.mu,self.cov):  
                #     pred = multivariate_normal(mean=m,cov=c).pdf(pix_vec)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(pix_vec) for mean,cov in zip(self.mu,self.cov)])
                #     prediction.append(pred)
                a = np.argmax(prediction) # a = 0,1,2
                seg_regs[x,y] = a              
                i += 1
                # print(i, prediction)
        print(seg_regs[10:50, :10])
        count = (seg_regs == 0).sum()
        print('count 0', count)
        count = (seg_regs == 1).sum()
        print('count 1', count)
        count = (seg_regs == 2).sum()
        print('count 2', count)

        # fig4 = plt.figure(figsize=(10,10))
        plt.imshow(seg_regs, interpolation='nearest')
        plt.show()


def create_input():
    w, h = 50, 50
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:25, 0:15] = [255, 0, 0] # red patch in upper left
    data[25:30, 0:15] =  [120, 200, 0]
    data[0:25, 30:45] =  [120, 20, 210]
    img = Image.fromarray(data, 'RGB')
    img.save('my.png')
    img.show()

def main():
    # create_input()
    # X = get_pixel2()
    # X,y = make_blobs(cluster_std = 1.5, random_state = 20, n_samples = 500, n_features=3, centers =3)

    X =  get_pixel2('star.png')
    gmm = GMM(X,3,50)
    gmm.run()
    print('X', type(X[20][0]), X[65])
    gmm.predict([X[10]], plot = True)
    # gmm.predict([[1,0,0]])
    # gmm.predict([[1/3,1/3,1/3]])
    # gmm.predict([[0.2,0.5,0.3]])
    # gmm.plot_segment('image_100.jpg')
    gmm.plot_segment('star.png')



if __name__ == '__main__':
    main()
