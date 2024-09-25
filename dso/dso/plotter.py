import matplotlib.pyplot as plt
from graphviz import Digraph
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rc,rcParams
# plt.style.use(['seaborn'])
# plt.style.use(['ggplot'])
plt.rcParams["ytick.minor.visible"]=False
plt.rcParams["xtick.minor.visible"]=False
plt.rcParams["axes.spines.left"]=True
plt.rcParams["axes.spines.bottom"]=True
plt.rcParams["axes.spines.top"]=False
plt.rcParams["axes.spines.right"]=False
plt.rcParams["font.family"]=['Arial']
# plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams["mathtext.fontset"]='stix'
# plt.rcParams["font.weight"] = "bold" 
# plt.rcParams['font.serif']=['SimHei']
# plt.rcParams["mathtext.fontset"]='Camt'
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['font.size']=16
# plt.rcParams['xtick.labelsize'] = 14
# plt.rcParams['ytick.labelsize'] = 14


class ExpressionTree:
    def __init__(self, preorder, arity):
        self.preorder = preorder
        self.arity = arity
        self.index = 0

    def build_tree(self):
        if self.index >= len(self.preorder):
            return None
        node_value = self.preorder[self.index]
        self.index += 1
        
        num_children = self.arity.get(node_value, 0)
        children = [self.build_tree() for _ in range(num_children)]
        return (node_value, *children)

    def draw_tree(self, node, graph, parent=None, counter=[0]):
        if node is None:
            return
        node_value = node[0]
        unique_node_id = f"{node_value}_{counter[0]}"
        counter[0] += 1
        graph.node(unique_node_id, label=node_value)

        if parent:
            graph.edge(parent, unique_node_id)

        for child in node[1:]:
            self.draw_tree(child, graph, unique_node_id, counter)

def tree_visualization(preorder, arity):
    tree = ExpressionTree(preorder, arity)
    root = tree.build_tree()

    graph = Digraph()
    tree.draw_tree(root, graph)
    graph.render('expression_tree', format='png', cleanup=True)
    return graph

# preorder_traversal = ['+', '-', 'x', 'a', 'b', 'c']
# operator_arity = {'+': 2, '-': 2, 'x': 2}  # Define arity for each operator
# graph = tree_visualization(preorder_traversal, operator_arity)

class Plotter:

    def __init__(self):
        pass



    def tree_plot(self, program):
        name_arites = program.library.name_arites
        preorder_traversal = [str(s) for s in program.traversal]
        graph = tree_visualization(preorder_traversal, name_arites)
        return graph  



    def evolution_plot(self,rewards):

        # print(rewards.shape)
        # print(rewards[:10])
        x = np.arange(1, len(rewards)+1)
        r_max = [np.max(r) for r in rewards]
        r_sub_avg = [np.mean(r) for r in rewards]
        r_best = np.maximum.accumulate(np.array(r_max))
        fig,axes = plt.subplots(figsize=(5,3))

        axes.plot(x,r_best, linestyle = '-',color= 'black',label ='Best')
        axes.plot(x,r_max, color='#F47E62',linestyle='-.', label = 'Maximum')
        axes.plot(x,r_sub_avg,color='#BD8EC0',linestyle='--', label = 'Mean of Top $\epsilon$')
        # axes.set_ylim(y_limit)
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Reward')

        # axes.xaxis.set_tick_params(top='off')#, direction='out', width=1)
        # axes.yaxis.set_tick_params(right='off')
        axes.legend(loc='best',frameon=False)
        # plt.savefig(out_path,dpi =dpi,bbox_inches='tight')
        plt.show()

    def density_plot(self, r_all, epoches = None):
    
        f, ax = plt.subplots(figsize=(5,3))
    
        colors = ['#0000a4', '#77007d','#a60057','#cb002f']
        colors = ['#0038c1', '#912e8c','green','#f1001c']
        if epoches is None:
            epoches = list(range(len(r_all)))
            palette1 = sns.light_palette("black", n_colors=len(epoches))
            cmap1= sns.light_palette("black", n_colors=len(epoches), as_cmap=True)
            palette2 = sns.light_palette("seagreen", n_colors=len(epoches))
            cmap2= sns.light_palette("seagreen", n_colors=len(epoches), as_cmap=True)
            cmaps, palettes=[cmap1,cmap2,cmap1], [palette1,palette2,palette1]

            sm = plt.cm.ScalarMappable( cmap=cmaps[1])
            palette = palettes[1]
            shade=False
            for i in range(len(epoches)):
                sns.kdeplot(r_all[epoches[i]], shade = shade, color = palette[i])
            cbar = f.colorbar(sm, ticks=[ 0,0.5,1],ax = ax)
            # cbar.ax.set_yticks([0,len(epoches)])
            cbar.ax.set_yticklabels([1,'',len(epoches)])  # vertically oriented colorbar
            ax.set_xlabel('Reward')   
            ax.set_ylabel('Density')   
        else:
            for i in range(len(epoches)):
                shade=True
                sns.kdeplot(r_all[epoches[i]], color=colors[i],clip=(0.0,1.0), label = f'Epoch={epoches[i]}', shade = shade)
            ax.set_xlabel('Reward')   
            ax.set_ylabel('Density')
            ax.legend( loc = 'best', frameon=False)
        plt.show()     
        # ax.set_xlim(0.4, 1.0)
        # plt.savefig(save_path+'.png', bbox_inches='tight', dpi = 600)
        # plt.savefig(save_path+'.pdf', bbox_inches='tight')
