import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_,_sig_ ,protdiv
import pandas as pd
import numpy as np
from sge.utilities.evaluations import *
from sge.utilities.optimizer import *
from scipy.optimize import minimize


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class SymbolicRegression():
    def __init__(self, function="quarticpolynomial", has_test_set=False, invalid_fitness=9999999):
        self.__train_set = []
        self.__test_set = None
        self.__number_of_variables = 1
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.Random()
        self.function = function
        self.has_test_set = has_test_set
        self.readpolynomial()
        self.calculate_rrse_denominators()
    def calculate_rrse_denominators(self):
        self.__RRSE_train_denominator = 0
        self.__RRSE_test_denominator = 0
        train_outputs = [entry[-1] for entry in self.__train_set]
        train_output_mean = float(sum(train_outputs)) / len(train_outputs)
        self.__RRSE_train_denominator = sum([(i - train_output_mean)**2 for i in train_outputs])
        if self.__test_set:
            test_outputs = [entry[-1] for entry in self.__test_set]
            test_output_mean = float(sum(test_outputs)) / len(test_outputs)
            self.__RRSE_test_denominator = sum([(i - test_output_mean)**2 for i in test_outputs])

    def read_fit_cases(self):
        f_in = open(self.__file_problem,'r')
        data = f_in.readlines()
        f_in.close()
        fit_cases_str = [ case[:-1].split() for case in data[1:]]
        self.__train_set = [[float(elem) for elem in case] for case in fit_cases_str]
        self.__number_of_variables = len(self.__train_set[0]) - 1

    def readpolynomial(self):
        def quarticpolynomial(inp):
            return pow(inp,4) + pow(inp,3) + pow(inp,2) + inp

        def kozapolynomial(inp):
            return pow(inp,6) - (2 * pow(inp,4)) + pow(inp,2)

        def pagiepolynomial(inp1,inp2):
            return 1.0 / (1 + pow(inp1,-4.0)) + 1.0 / (1 + pow(inp2,-4))

        def keijzer6(inp):
            return sum([1.0/i for i in range(1,inp+1,1)])

        def keijzer9(inp):
            return _log_(inp + (inp**2 + 1)**0.5)

        if self.function in ["pagiepolynomial"]:
            function = eval(self.function)
            # two variables
            l = []
            for xx in drange(-5,5.4,0.4):
                for yy in drange(-5,5.4,0.4):
                    zz = pagiepolynomial(xx,yy)
                    l.append([xx,yy,zz])

            self.__train_set=l
            self.training_set_size = len(self.__train_set)
            if self.has_test_set:
                xx = list(drange(-5,5.0,.1))
                yy = list(drange(-5,5.0,.1))
                function = eval(self.function)
                zz = map(function, xx, yy)

                self.__test_set = [xx,yy,zz]
                self.test_set_size = len(self.__test_set)
        elif self.function in ["quarticpolynomial"]:
            function = eval(self.function)
            #l = []
            #for xx in drange(-1,1.1,0.1):
            #    yy = quarticpolynomial(xx)
            #    l.append([xx,yy])

            #self.__train_set = l
            df = pd.read_csv('dataset/Synthetic/alerceZTFv7.1_4spm-mcmc_352_red.txt', sep=",",header=None)
            l=df.to_numpy().tolist()
            self.__train_set =l
            self.training_set_size = len(self.__train_set)
            if self.has_test_set:
                #xx = list(drange(-1,1.1,0.1))
                #function = eval(self.function)
                #yy = map(function, xx)
                print('Leyendo test de proyecto....')
                df = pd.read_csv('dataset/Synthetic/alerceZTFv7.1_4spm-mcmc_352_red.txt', sep=",",header=None)
                l=df.to_numpy().tolist()
                self.__test_set = l
                self.test_set_size = len(self.__test_set)
        else:
            if self.function == "keijzer6":
                xx = list(drange(1,51,1))
            elif self.function == "keijzer9":
                xx = list(drange(0,101,1))
            else:
                xx = list(drange(-1,1.1,.1))
            function = eval(self.function)
            yy = map(function,xx)
            self.__train_set = list(zip(xx, yy))
            self.__number_of_variables = 1
            self.training_set_size = len(self.__train_set)
            if self.has_test_set:
                if self.function == "keijzer6":
                    xx = list(drange(51,121,1))
                elif self.function == "keijzer9":
                    xx = list(drange(0,101,.1))
                yy = map(function,xx)
                self.__test_set = [xx, yy]
                self.test_set_size = len(self.__test_set)

    def get_error(self, individual, dataset):
        pred_error = 0
        cuociente = 0
        maxData=0
        posData=0
        maxPred=0
        posPred=0;
        kij=1
        size=0
        result_array=np.array([])
        x_array=np.array([])
        y_array=np.array([])

        g1=lambda t: str_to_value(t,individual)

        for fit_case in dataset:
            case_output = fit_case[-1]
            try:
                result = g1(x_array)
                result2 = eval(individual, globals(), {"x": fit_case[:-1]})
                x_array=np.append(x_array,fit_case[:-1])
                y_array=np.append(y_array,fit_case[-1])
                result_array=np.append(result_array,result2)


            except (OverflowError, ValueError) as e:
                return self.__invalid_fitness

        g1=lambda t: str_to_value(t,individual)
        g1=lambda t: eval(individual,globals(),{"x":t})
        output=g1(x_array)

        #RMSE=root_mean_squared_error(y_array, output)
        RMSE=root_mean_squared_error(y_array, result_array)
        return RMSE

    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None

        error = self.get_error(individual, self.__train_set)
        #error = _sqrt_( error)

        #error = _sqrt_( error/ self.__RRSE_train_denominator)

        if error is None:
            error = self.__invalid_fitness

        if self.__test_set is not None:
            test_error = 0
            test_error = self.get_error(individual, self.__test_set)
            test_error = _sqrt_( test_error)
            #test_error = _sqrt_( test_error / float(self.__RRSE_test_denominator))

        return error, {'generation': 0, "evals": 1, "test_error": test_error}


if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func, parameters_file="parameters/standardv3.yml")
