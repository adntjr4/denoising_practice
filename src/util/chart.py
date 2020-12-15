import asciichartpy


class LossChart():
    def __init__(self):
        self.color = [asciichartpy.lightgreen, asciichartpy.lightred, asciichartpy.lightmagenta]
    
        self.compression_number  = [1,5,None]

        self.data = []
        self.plot_data = [[], [], []]

    def add_data(self, f_num):
        self.data.append(f_num)

        data_len = len(self.data)

        for i in range(3):
            if self.compression_number[i] is not None:
                if data_len % self.compression_number[i] == 0:
                    t_data = self.data[-self.compression_number[i]:]
                    self.plot_data[i].append(sum(t_data)/len(t_data))

    def return_chart(self, height, width):
        return asciichartpy.plot([serial[-width:] for serial in self.plot_data if serial != []], {'height':height, 'colors':self.color, 'format':'{:10.4e}'})
