from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import stock_names
from .models import stocks_data
from django.http import HttpResponse, JsonResponse
from django.template import RequestContext
from io import BytesIO
import base64
from io import StringIO
from pylab import *
import json

# Create your views here.
def index(request):
    return render(request, 'blog/index.html', {})
def home(request):
    return render(request, 'blog/test.html', {})
def listed_companies(request):
    item = stock_names.get_all_names()
    return render(request, 'blog/listed_companies.html', {'item': item})

def analysis(request):
    item = stock_names.get_all_names()
    item_list = item
    item = json.dumps(item)
    _dict = {
        'item': item,
        'item_list': item_list
    }
    return render(request, 'blog/analysis.html', _dict)
def projects(request):
    return render(request, 'blog/projects.html', {})
def contact(request):
    return render(request, 'blog/contact.html', {})


def simple(request, stock_name):

    from matplotlib import pylab, cm
    import matplotlib.pyplot as plt
    import PIL, PIL.Image
    import pandas as pd
    import numpy as np
    from pandas.tools.plotting import scatter_matrix

    if request.method == 'POST':
        #Company wise plot data
        company_stock_name = request.POST['company_stock_name']
        first = request.POST['first']
        second = request.POST['second']
        company_plot_kind = request.POST['company_plot_kind']

        #Comparison plot data
        stock_name_first = request.POST['stock_name_first']
        stock_name_second = request.POST['stock_name_second']
        Column = request.POST['Column']
        plot_kind = request.POST['plot_kind']

        #Check box data
        check = request.POST['check']

    if check == '1': #check if company wise plot was selected

        #define parameters
        cols = [first, second]

        start_date = None
        end_date = None

        # Construct the graph
        #t = arange(0.0, 2.0, 0.01)
        #s = sin(2*pi*t)
        #plot(t, s, linewidth=1.0)
        #cols=['Closing Price', 'Maximum Price']
        #start_date=None
        #end_date=None

        item = stock_names.get_all_names()
        filename = '/home/sankalpa/Major_project/data/' + company_stock_name + '.csv'
        df = pd.read_csv(filename, index_col=0, parse_dates=True)

        if(company_plot_kind == 'hexbin'):
        #hexbin plot
            df.ix[:,cols][start_date:end_date].plot(kind='hexbin', x=cols[0], y=cols[1], gridsize=25)
        if(company_plot_kind == 'scatter_matrix'):
        #scatter matrix plot
            scatter_matrix(df.ix[:,cols][start_date:end_date], alpha=0.2, diagonal='kde')
        if(company_plot_kind == 'line'):
        #line plot
            df.ix[:,cols][start_date:end_date].plot(kind=company_plot_kind, subplots=True)
        #ohlc
        if(company_plot_kind == 'ohlc'):
            import cufflinks as cf
            import plotly
            import plotly.offline as py
            from plotly.offline.offline import _plot_html
            import plotly.graph_objs as go
            from plotly.tools import FigureFactory as FF
            fig = FF.create_candlestick(df['Opening Price'], df['Maximum Price'], df['Minimum Price'], df['Closing Price'], dates=df.index)
            py.plot(fig,filename=filename+'ohlc', validate=False)
        #macd
        if(company_plot_kind == 'macd'):
            import cufflinks as cf
            import plotly
            import plotly.offline as py
            from plotly.offline.offline import _plot_html
            import plotly.graph_objs as go
            from plotly.tools import FigureFactory as FF
            fig = df['Closing Price'].ta_plot(study='macd', fast_period=12, slow_period=26, signal_period=9, asFigure=True)
            py.plot(fig,filename=filename+'macd',validate=False)


    if check == '2': #check if comparison plot was selected

        name = [stock_name_first, stock_name_second]
        cols = [Column]
        start_date = None
        end_date = None
        filenames = ['/home/sankalpa/Major_project/cleaneddata/' + company + '.csv' for company  in name]
        try:
            data = pd.concat([pd.read_csv(company, index_col=0, parse_dates=True) for company in filenames], axis=1, keys=name)
        except(FileNotFoundError, IOError):
            print('Wrong file or file path.')
            return

        #if plot_kind = line or box
        ax = data.ix[:, data.columns.get_level_values(1).isin(cols)][start_date:end_date].plot()
        ax.set_title('{} Plot of {} of {}.'.format(plot_kind.title(),','.join(cols), ','.join(name)))
        plt.legend(title='Companies', fancybox=True, shadow=True, loc='best')


        #plt.show()

    # Store image in a string buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    data = buffer.getvalue()
    data = base64.b64encode(data)
    text_file = open("/home/sankalpa/output.txt", "wb")
    text_file.write(data)
    text_file.close()
    return HttpResponse(data, content_type='text/plain')




def listing(request, stock_name):
    item = stock_names.get_all_names()
    #item = json.dumps(item)
    data = stocks_data.get_data(stock_name)
    paginator = Paginator(data, 50)

    page = request.GET.get('page')
    try:
        current_data = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        current_data = paginator.page(1)
    except EmptyPage:
        # If page is out of range, deliver last page.
        current_data = paginator.page(paginator.num_pages)

    # Calculate the index of stock_name
    index = 0
    for i in item:
        if i[1] != stock_name:
            index = index + 1;
        else:
            break;

    item_list = item
    item = json.dumps(item)
    _dict = {
        'current_data': current_data,
        'data': data,
        'item': item,
        'index': index,
        'item_list': item_list
    }

    #return render(request, 'blog/stocks/' + stock_name + '.html', _dict)
    return render(request, 'blog/base_stock.html', _dict)

def train (request):
    import logging
    import os.path
    import sys
    sys.path.insert(0, '/home/sankalpa/simple-blog/mysite/blog/templates/blog/analysis')
    from prediction import prepareInput as pi
    from preprocessing import moreIndicators as indi
    from logger import log as log
    from pybrain.utilities import percentError
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.supervised.trainers import BackpropTrainer
    from pybrain.structure.modules import SoftmaxLayer, TanhLayer, LinearLayer, SigmoidLayer
    from pybrain.tools.customxml.networkwriter import NetworkWriter
    from pybrain.tools.customxml.networkreader import NetworkReader
    from pybrain.tools.validation import ModuleValidator, CrossValidator
    from sklearn.metrics import accuracy_score, precision_score, classification_report, roc_curve, auc
    import matplotlib.pyplot as plt

    if request.method == 'POST':
        # data
        stock_name = request.POST['stock_name']
        features = request.POST['features']
        window_size = request.POST['window_size']
        proportion = request.POST['proportion']
        nNeurons = request.POST['nNeurons']
        nHorizon = request.POST['nHorizon']

    input_list = []
    features_list = ['EMA', 'Momentum', 'RSI']

    for feature in features_list:
        if feature in features:
            input_list.append(feature)


    log.setup_logging()
    logger = logging.getLogger()
    logger.info('+++++++++++++++++++++++++')
    logger.info('file: train.py')

    logger.info('init logging..')

    def load_dataset(stockname, window):
        '''
        load dataset from csv filename

        Parameters:
        stockname: name of the csv file of stock data

        Returns:
        dataframe: pandas dataframe
        '''
        print('loading dataset...')
        # load dataframe from csv file
        dataframe = pi.load_data_frame(stockname)
        # change datetime index to integer index
        # TODO: manage issue with datetime index
        dataframe = pi.signal_updown(dataframe, window)
        dataframe.index = range(len(dataframe.index))
        # change column name to match with indicator calculating module
        dataframe.columns = [
            'Transactions',
            'Traded_Shares',
            'Traded_Amount',
            'High',
            'Low',
            'Close',
            'signal']
        return dataframe

    def select_features(dataframe, n, prop, features):
        '''
        select input and output features to prepate training and testing dataset

        Parameters:
        dataframe: dataframe of the stock data

        Returns:
        trndata: training dataset
        tstdata: testing dataset
        '''
        #TODO: dynamic feature selection
        print('selecting features...')
        # calculate and add indicators to dataframe
        dataframe = indi.EMA(dataframe, n)
        dataframe = indi.RSI(dataframe, n)
        dataframe = indi.MOM(dataframe, n)
        # prepate dataset for training and testing
        #input_features = ['RSI_' + str(n), 'Momentum_'+str(n)] #'EMA_'+str(n),
        input_features = [feature+'_'+str(n) for feature in features]
        output_features = ['signal']
        ds, trndata, tstdata = pi.prepare_datasets(
            input_features, output_features, dataframe[n:], prop)

        print('input features: ' + str(input_features)  +', Output : ' + str(output_features))
        logger.info(
            'input features: ' +
            str(input_features) +
            ', Output : ' +
            str(output_features))
        return ds, trndata, tstdata

    def build_network(trndata, tstdata, hidden_dim, fout, load_network):
        '''
        build ANN network with backpropagation trainer

        Parameters:
        trndata: training dataset
        tstdata: testing dataset
        hidden_dim: no of neuron in hidden layer
        fout: filename to load saved network
        load_network: True: load network, False: dont load network

        Returns:
        fnn: feedforward neural network
        trainer: back propagation trainer
        '''
        # build network
        print('creating neural network...')
        logger.info('creating neural network')
        # if(os.path.isfile('ann.xml') and load_network == 1):
        #     fnn = NetworkReader.readFrom('ann.xml')
        # else:
        fnn = buildNetwork(trndata.indim,hidden_dim,trndata.outdim,hiddenclass = SigmoidLayer, outclass=SoftmaxLayer)
        print('neural network:\n'+str(fnn)+'\ninput_dim ='+str(trndata.indim)+', hidden_dim=' + str(hidden_dim) + ', output_dim ='+str(trndata.outdim))
        print('creating backprop Trainer...')
        logger.info('creating backprop Trainer')

        # set up brckprop trainer
        trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.01, verbose=True, weightdecay=0.01)
        return fnn, trainer

    def train_network(trndata,tstdata,fnn,trainer):
        '''
        train the network using trainer

        Parameters:
        trndata: training dataset
        tstdata: testing dataset
        fnn: feed forward NN
        trainer: backprop trainer
        '''
        print('training and testing network')
        logger.info('training and testing network')
        # # start training iterations
        # for i in range(10):
        #     error = trainer.trainEpochs(1)
        #     # cv = CrossValidator( trainer, trndata, n_folds=5)
        #     # CrossValidator.validate(cv)
        #     # print("MSE %f @ %i" %( cv.validate(), i ))
        #     trnresult = percentError(trainer.testOnClassData(),trndata['class'])
        #     tstresult = percentError(trainer.testOnClassData(dataset = tstdata),tstdata['class'])
        #     print("epoch: %4d"%trainer.totalepochs,"\ntrain error: %5.2f%%"%trnresult,"\ntest error: %5.2f%%"%tstresult)

        # train the network until convergence
        trainer.trainUntilConvergence(verbose=True,
                                    trainingData=trndata,
                                    validationData=tstdata,
                                    maxEpochs=5)
        # NetworkWriter.writeToFile(fnn,'ann.xml')

    def activate_network(ds, tstdata, fnn, nhorizon):
        print('activating network on data')
        # activate network for test data
        out = fnn.activateOnDataset(tstdata)
        print(len(out))
        # index of  maximum value gives the class
        predictions = out.argmax(axis=1)
        print("The Result for",nhorizon,"day ahead :")
        nxt = fnn.activate(ds)
        print(nxt, 'up' if nxt.argmax(axis=0) else 'down')
        actual = tstdata['target'].argmax(axis=1)
        temp = []
        for i in range(len(actual)):
            temp.append((actual[i],predictions[i]))
        # result analysis, uses scikitlearn metrics
        target_names = ['up', 'down']
        accuracy = accuracy_score(tstdata['target'].argmax(axis=1), predictions)
        precision = precision_score(tstdata['target'].argmax(axis=1), predictions)
        if(nxt.argmax(axis=0)):
            trend = 'UP'
        else:
            trend = 'DOWN'
        classification_result = classification_report(tstdata['target'].argmax(axis=1), predictions, target_names=target_names)
        print(classification_result)
        my_list = classification_result.split()
        print(my_list)
        # words to remove
        my_list.remove('precision')
        my_list.remove('recall')
        my_list.remove('f1-score')
        my_list.remove('support')
        my_list.remove('up')
        my_list.remove('down')
        my_list.remove('avg')
        my_list.remove('total')
        my_list.remove('/')

        print(my_list)
        print('Result on testdata')
        print(classification_report(tstdata['target'].argmax(axis=1),predictions,target_names=target_names))
        print('accuracy= ', accuracy_score(tstdata['target'].argmax(axis=1), predictions))
        # The precision is the ratio tp / (tp + fp)
        print('precision= ', precision_score(tstdata['target'].argmax(axis=1), predictions))
        # logger.info('\n' + classification_report(tstdata['target'].argmax(axis=1),predictions,target_names=target_names))
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.title('Receiver Operating Characteristic')
        # plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
        # plt.legend(loc='lower right')
        # plt.plot([0,1],[0,1],'r--')
        # plt.xlim([-0.1,1.2])
        # plt.ylim([-0.1,1.2])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
        nxt = ','.join([ str(i) for i in nxt])
        my_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'my_list': my_list,
            'next': nxt,
            'trend': trend
        }
        return my_dict

    def ann(csvname, window, prop, neurons, features = ['RSI','Momentum'], nhorizon = 1):
        # n = 20
        # prop = 0.20
        # set up logger
        dataframe = load_dataset(csvname, nhorizon)
        # print(dataframe.describe())
        # print(dataframe[-5:])
        ds, trndata, tstdata = select_features(dataframe, window, prop, features)
        #print(ds.calculateStatistics())
        #print(len(ds),len(trndata))
        predict = []
        if(ds.indim == 1):
            predict = ds['input'][-1:]
        else:
            predict = ds['input'][:][-1:][0]
        fnn, trainer = build_network(trndata, tstdata, neurons, 'ann.xml', 0)
        train_network(trndata, tstdata, fnn, trainer)
        result = activate_network(predict,tstdata, fnn, nhorizon)
        return result

    #if __name__ == '__main__':
    result = ann('/home/sankalpa/Major_project/cleaneddata/' + stock_name + '.csv', int(window_size), float(proportion), int(nNeurons), features = input_list, nhorizon = 1)
    #result = json.dumps(result)
    #value = ','.join([ str(i) for i in result])
    #return HttpResponse(result)
    return JsonResponse(result, content_type='text/plain', safe=False)

# ## weights of connections
# print('weights of connections')
# ## input layer
# print(fnn['in'].outputbuffer[fnn['in'].offset])
# ## hidden layer
# print(fnn['hidden0'].outputbuffer[fnn['hidden0'].offset])


