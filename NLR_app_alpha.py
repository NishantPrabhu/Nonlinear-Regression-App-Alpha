
# Dependencies
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import streamlit as st 

# Header
st.title('Univariate Nonlinear Regression')
st.write("This app has been made primarily for solving problems in **Inverse Methods for Heat Transfer**. You can adjust the parameters and choose the algorithm based on the problem you are solving; the app will perform computations and provide you some parameters to see how well it fits your data (visual and statistical). Or, you can upload a CSV file containing your data with the specifications given below and let the app do the rest.")

data_input = st.radio('How will you provide the data?', \
                       ['Parameter Input', 'Data Upload'], 0)

# Take action based on data_input
if data_input == 'Parameter Input':

    st.write('## **Parameter Input**')
    st.write('This feature is here for experimentation and learning. It will estimate the parameters for the following equation, arising out of unsteady heating problem with convection.')
    r'''
    $$
    mC_{p}\frac{dT}{dt}=Q-hA(T-T_{\infty})\quad\Rightarrow\quad \boxed{T-T_{\infty}=\theta=\frac{Q}{hA}\left(1-e^{-\frac{hA}{mC_{p}}t}\right)}
    $$
    '''
    st.write('To keep it simple, we will reduce this to two parameters as follows.')
    r'''
    $$
    \frac{Q}{hA}=a \quad;\quad\frac{hA}{mC_{p}}=b\quad\Rightarrow\quad \boxed{\theta=a\left(1-e^{-bt}\right)}
    $$
    '''
    st.write('')
    r'''Below, input the values of $a$ and $b$ that you want. Also, provide the upper and lower limits for time $t$ and the number of sample points you want between them.'''

    a = st.number_input("Value of $a$ (between 10.0 and 200.0)", 10.0, 200.0, 40.0)
    b_ = st.number_input("Value of $100b$ (between 0.01 and 10)", 0.01, 10.0, 0.5)
    t_llim = st.number_input("Value of starting time in seconds (between 0 and 500)", 0, 500, 0)
    t_rlim = st.number_input("Value of ending time in seconds (between 0 and 100) and greater than starting time", 0, 500, 100)

    if t_llim >= t_rlim:
        raise Exception('Left limit of t should be strictly less than right limit of t.')

    n_samples = st.number_input("How many sample points do you want? (between 2 and 1000)", 2, 1000, 100)
    noise_var = st.slider("Noise variance (mean = 0.0)", 0.0, 10.0, 3.0)

    b = b_ * 0.01

    # Generate data
    t = np.linspace(t_llim, t_rlim, n_samples)
    T_real = a * (1.0 - np.exp(-b*t)).reshape((-1, 1))

    noise = np.random.normal(loc=0, scale=noise_var, size=(T_real.shape[0])).reshape((-1, 1))
    T_noisy = (T_real + noise).reshape((-1, 1))

    # Value and Jacobian functions
    def F(t, a, b):
        """The function governing temperature of the body"""
        return (a*(1-np.exp(-b*t))).reshape((-1, 1))

    def J_a(t, a, b):
        """Gradient of the function w.r.t. parameter a"""
        return (1 - np.exp(-b*t)).reshape((-1, 1))

    def J_b(t, a, b):
        """Gradient of the function w.r.t. parameter b"""
        return (a*t*np.exp(-b*t)).reshape((-1, 1))

    # Initial guess
    st.write('In the section below, provide your initial guess for the parameter values.')
    init_a = st.number_input("Initial guess for $a$", 10.0, 200.0, a)
    init_b_ = st.number_input("Initial guess for $100b$", 0.01, 10.0, 100*b)
    init_b = init_b_ * 0.01

    method = st.radio("Choose computation method", \
                       ['Gauss-Newton Algorithm', 'Levenberg Algorithm', 'Levenberg-Marquardt Algorithm'], 0)

    # Initialize matrices
    A = np.array([[init_a],      # <-- a
                  [init_b]])     # <-- b

    # Forcing vector
    D = np.zeros((T_noisy.shape[0], 1))

    # Jacobian matrix
    z = np.zeros((T_noisy.shape[0], A.shape[0]))

    # Update vector
    dA = np.zeros((A.shape[0], 1))
    
    # For each method
    if method == 'Gauss-Newton Algorithm':

        st.write("## Gauss Newton Algorithm")
        r'''
        $$
        \Delta A=\eta \cdot \left(Z^{T}Z\right)^{-1}\left(Z^{T}D\right)
        $$
        '''
        eta = st.slider("Choose value of $\eta$. This parameter suppresses the magnitudes of updates to prevent divergence.", 0.01, 1.00, 0.30)

        # Perform iterations
        a_estimates = []
        b_estimates = []         

        for _ in range(100):
            D = (T_noisy - F(t, A[0], A[1])) * eta
            z = np.hstack((J_a(t, A[0], A[1]), J_b(t, A[0], A[1])))
            dA = np.dot(np.linalg.pinv(np.dot(z.T, z)), np.dot(z.T, D)) 
            A += dA
            a_estimates.append(A[0][0])
            b_estimates.append(A[1][0])

    elif method == 'Levenberg Algorithm':

        st.write("## Levenberg Algorithm")
        r'''
        $$
        \Delta A=\eta \cdot \left(Z^{T}Z + \lambda I\right)^{-1}\left(Z^{T}D\right)
        $$
        '''
        eta = st.slider("Choose value of $\eta$. This parameter suppresses the magnitudes of updates to prevent divergence.", 0.01, 1.00, 0.30)

        lbd_ = st.slider("Choose value of $10\lambda$. This parameter helps improve the model's robustness.", 0.00, 10.00, 0.03)

        lbd = lbd_ * 0.1

        # Perform iterations
        a_estimates = []
        b_estimates = []         

        for _ in range(100):
            D = (T_noisy - F(t, A[0], A[1])) * eta
            z = np.hstack((J_a(t, A[0], A[1]), J_b(t, A[0], A[1])))
            zt_z = np.dot(z.T, z)
            dA = np.dot(np.linalg.pinv(zt_z + lbd * np.identity(zt_z.shape[0])), np.dot(z.T, D)) 
            A += dA
            a_estimates.append(A[0][0])
            b_estimates.append(A[1][0])

    elif method == 'Levenberg-Marquardt Algorithm':

        st.write("## Levenberg-Marquardt Algorithm")
        r'''
        $$
        \Delta A=\eta \cdot \left(Z^{T}Z + \lambda \cdot diag\left(Z^{T}Z\right) \right)^{-1}\left(Z^{T}D\right)
        $$
        '''
        eta = st.slider("Choose value of $\eta$. This parameter suppresses the magnitudes of updates to prevent divergence.", 0.01, 1.00, 0.30)

        lbd_ = st.slider("Choose value of $10\lambda$. This parameter helps improve the model's robustness.", 0.00, 10.00, 0.03)

        lbd = lbd_ * 0.1

        # Perform iterations
        a_estimates = []
        b_estimates = []         

        for _ in range(100):
            D = (T_noisy - F(t, A[0], A[1])) * eta
            z = np.hstack((J_a(t, A[0], A[1]), J_b(t, A[0], A[1])))
            zt_z = np.dot(z.T, z)
            dA = np.dot(np.linalg.pinv(zt_z + lbd * np.identity(zt_z.shape[0]) * zt_z), np.dot(z.T, D)) 
            A += dA
            a_estimates.append(A[0][0])
            b_estimates.append(A[1][0])

    st.write("### **Regression Results**")
    st.write("Estimated parameter values")
    st.write(pd.DataFrame(
        np.array([['Value of a', a, a_estimates[-1]],
                    ['Value of b', b, b_estimates[-1]]]),
        columns=['Description', 'Actual value', 'Estimated value']
    ))

    st.write('')
    # Plot estimation trend
    if st.checkbox("Show estimation trends", value=False):
        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot(121)
        ax1.plot(a_estimates, color='red', linewidth=3)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Estimate of a')
        ax1.set_title('a estimates')
        ax1.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(b_estimates, color='blue', linewidth=3)
        ax2.set_title('b estimates')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Estimate of b')
        ax2.grid()

        plt.tight_layout()
        st.pyplot(fig)

    if st.checkbox("Show fit curve", value=False):
        # Visualizing the fit line
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(t, T_noisy, color='red', marker='o', label='data')
        x_vals = np.linspace(t_llim, t_rlim, 1000)
        fit_vals = a_estimates[-1]*(1 - np.exp(-b_estimates[-1]*x_vals))
        ax.plot(x_vals, fit_vals, color='black', linewidth=3, label='fit line')
        plt.title('Fit line')
        plt.xlabel('t (s)')
        plt.ylabel('Theta (K)')
        plt.legend()
        plt.grid()
        st.pyplot(fig)

    if st.checkbox("Show parity plot", value=False):
        # Parity plot
        predicted_T = a_estimates[-1]*(1-np.exp(-b_estimates[-1]*t))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(predicted_T, T_noisy, c='red', alpha=0.6)
        lim = np.linspace(*ax.get_xlim())
        ax.plot(lim, lim, color='black')
        plt.xlabel('Predictions')
        plt.ylabel('Target')
        plt.title('Parity plot')
        plt.grid()
        st.pyplot(fig)

    if st.checkbox("Show residuals table", value=False):
        # Residuals table
        # Make a dataframe with x, T_noisy and predicted_T
        predicted_T = a_estimates[-1]*(1-np.exp(-b_estimates[-1]*t))
        data = np.vstack((t.reshape((1, -1)), T_noisy.reshape((1, -1)), predicted_T.reshape((1, -1)))).T
        df = pd.DataFrame(data, columns=['t', 'T_measured', 'T_fit'])

        df['Mean_deviation'] = (df['T_measured'] - df['T_measured'].mean())**2
        df['Fit_deviation'] = (df['T_measured'] - df['T_fit'])**2

        st.write(df)

        # Calculate goodness of fit parameters
        S_t, S_r = df['Mean_deviation'].sum(), df['Fit_deviation'].sum()
        R_2 = (S_t - S_r)/S_t
        SE = np.sqrt(S_r/(len(t) - 2))

        st.write('')
        st.write('Goodness of fit parameters')
        st.write(pd.DataFrame(
            np.array([['Coefficient of determination', R_2],
                    ['Correlation coefficient', R_2**0.5],
                    ['Standard error', SE]]),
            columns = ['Description', 'Value']
        ))

if data_input == 'Data Upload':

    st.write('## **Data Upload**')
    st.write("Upload your data as a CSV file with two columns named 'x' and 'target'. Please ensure that your data doesn't have any missing values. The app will perform regression and show you the results in a jiffy!")

    # Initial guess
    st.write('In the section below, provide your initial guess for the parameter values.')
    init_a = st.number_input("Initial guess for $a$", 10.0, 200.0, 40.0)
    init_b_ = st.number_input("Initial guess for $100b$", 0.01, 10.0, 1.00)
    init_b = init_b_ * 0.01

    method = st.radio("Choose computation method", \
                       ['Gauss-Newton Algorithm', 'Levenberg Algorithm', 'Levenberg-Marquardt Algorithm'], 0)

    up_file = st.file_uploader("Upload file", type="csv")

    if up_file is not None:
        data = pd.read_csv(up_file)
        x = data['x'].values
        T_noisy = data['target'].values

        # Initialize matrices
        A = np.array([[init_a],      # <-- a
                    [init_b]])     # <-- b

        # Forcing vector
        D = np.zeros((T_noisy.shape[0], 1))

        # Jacobian matrix
        z = np.zeros((T_noisy.shape[0], A.shape[0]))

        # Update vector
        dA = np.zeros((A.shape[0], 1))
        
        # For each method
        if method == 'Gauss-Newton Algorithm':

            st.write("## Gauss Newton Algorithm")
            r'''
            $$
            \Delta A=\eta \cdot \left(Z^{T}Z\right)^{-1}\left(Z^{T}D\right)
            $$
            '''
            eta = st.slider("Choose value of $\eta$. This parameter suppresses the magnitudes of updates to prevent divergence.", 0.01, 1.00, 0.30)

            # Perform iterations
            a_estimates = []
            b_estimates = []         

            for _ in range(100):
                D = (T_noisy - F(t, A[0], A[1])) * eta
                z = np.hstack((J_a(t, A[0], A[1]), J_b(t, A[0], A[1])))
                dA = np.dot(np.linalg.pinv(np.dot(z.T, z)), np.dot(z.T, D)) 
                A += dA
                a_estimates.append(A[0][0])
                b_estimates.append(A[1][0])

        elif method == 'Levenberg Algorithm':

            st.write("## Levenberg Algorithm")
            r'''
            $$
            \Delta A=\eta \cdot \left(Z^{T}Z + \lambda I\right)^{-1}\left(Z^{T}D\right)
            $$
            '''
            eta = st.slider("Choose value of $\eta$. This parameter suppresses the magnitudes of updates to prevent divergence.", 0.01, 1.00, 0.30)

            lbd_ = st.slider("Choose value of $10\lambda$. This parameter helps improve the model's robustness.", 0.00, 10.00, 0.03)

            lbd = lbd_ * 0.1

            # Perform iterations
            a_estimates = []
            b_estimates = []         

            for _ in range(100):
                D = (T_noisy - F(t, A[0], A[1])) * eta
                z = np.hstack((J_a(t, A[0], A[1]), J_b(t, A[0], A[1])))
                zt_z = np.dot(z.T, z)
                dA = np.dot(np.linalg.pinv(zt_z + lbd * np.identity(zt_z.shape[0])), np.dot(z.T, D)) 
                A += dA
                a_estimates.append(A[0][0])
                b_estimates.append(A[1][0])

        elif method == 'Levenberg-Marquardt Algorithm':

            st.write("## Levenberg-Marquardt Algorithm")
            r'''
            $$
            \Delta A=\eta \cdot \left(Z^{T}Z + \lambda \cdot diag\left(Z^{T}Z\right) \right)^{-1}\left(Z^{T}D\right)
            $$
            '''
            eta = st.slider("Choose value of $\eta$. This parameter suppresses the magnitudes of updates to prevent divergence.", 0.01, 1.00, 0.30)

            lbd_ = st.slider("Choose value of $10\lambda$. This parameter helps improve the model's robustness.", 0.00, 10.00, 0.03)

            lbd = lbd_ * 0.1

            # Perform iterations
            a_estimates = []
            b_estimates = []         

            for _ in range(100):
                D = (T_noisy - F(t, A[0], A[1])) * eta
                z = np.hstack((J_a(t, A[0], A[1]), J_b(t, A[0], A[1])))
                zt_z = np.dot(z.T, z)
                dA = np.dot(np.linalg.pinv(zt_z + lbd * np.identity(zt_z.shape[0]) * zt_z), np.dot(z.T, D)) 
                A += dA
                a_estimates.append(A[0][0])
                b_estimates.append(A[1][0])

        st.write("### **Regression Results**")
        st.write("Estimated parameter values")
        st.write(pd.DataFrame(
            np.array([['Value of a', a, a_estimates[-1]],
                        ['Value of b', b, b_estimates[-1]]]),
            columns=['Description', 'Actual value', 'Estimated value']
        ))

        st.write('')
        # Plot estimation trend
        if st.checkbox("Show estimation trends", value=False):
            fig = plt.figure(figsize=(16, 6))

            ax1 = fig.add_subplot(121)
            ax1.plot(a_estimates, color='red', linewidth=3)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Estimate of a')
            ax1.set_title('a estimates')
            ax1.grid()

            ax2 = fig.add_subplot(122)
            ax2.plot(b_estimates, color='blue', linewidth=3)
            ax2.set_title('b estimates')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Estimate of b')
            ax2.grid()

            plt.tight_layout()
            st.pyplot(fig)

        if st.checkbox("Show fit curve", value=False):
            # Visualizing the fit line
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(t, T_noisy, color='red', marker='o', label='data')
            x_vals = np.linspace(t_llim, t_rlim, 1000)
            fit_vals = a_estimates[-1]*(1 - np.exp(-b_estimates[-1]*x_vals))
            ax.plot(x_vals, fit_vals, color='black', linewidth=3, label='fit line')
            plt.title('Fit line')
            plt.xlabel('t (s)')
            plt.ylabel('Theta (K)')
            plt.legend()
            plt.grid()
            st.pyplot(fig)

        if st.checkbox("Show parity plot", value=False):
            # Parity plot
            predicted_T = a_estimates[-1]*(1-np.exp(-b_estimates[-1]*t))
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(predicted_T, T_noisy, c='red', alpha=0.6)
            lim = np.linspace(*ax.get_xlim())
            ax.plot(lim, lim, color='black')
            plt.xlabel('Predictions')
            plt.ylabel('Target')
            plt.title('Parity plot')
            plt.grid()
            st.pyplot(fig)

        if st.checkbox("Show residuals table", value=False):
            # Residuals table
            # Make a dataframe with x, T_noisy and predicted_T
            predicted_T = a_estimates[-1]*(1-np.exp(-b_estimates[-1]*t))
            data = np.vstack((t.reshape((1, -1)), T_noisy.reshape((1, -1)), predicted_T.reshape((1, -1)))).T
            df = pd.DataFrame(data, columns=['t', 'T_measured', 'T_fit'])

            df['Mean_deviation'] = (df['T_measured'] - df['T_measured'].mean())**2
            df['Fit_deviation'] = (df['T_measured'] - df['T_fit'])**2

            st.write(df)

            # Calculate goodness of fit parameters
            S_t, S_r = df['Mean_deviation'].sum(), df['Fit_deviation'].sum()
            R_2 = (S_t - S_r)/S_t
            SE = np.sqrt(S_r/(len(t) - 2))

            st.write('')
            st.write('Goodness of fit parameters')
            st.write(pd.DataFrame(
                np.array([['Coefficient of determination', R_2],
                        ['Correlation coefficient', R_2**0.5],
                        ['Standard error', SE]]),
                columns = ['Description', 'Value']
            ))



