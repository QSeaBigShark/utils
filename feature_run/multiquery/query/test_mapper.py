import sys
sys.path.append('.')
import v03

if __name__ == '__main__':
    title = ['name', 'idcard', 'phone', 'loan_dt', 'label']
    name = sys.argv[1]
    fa = open(name, 'r')
    fn = open('v03.meta', 'r')
    fw = open(name+r'.feature', 'w')
    for s in title:
        fw.write(s+'\t')

    line_n = fn.readline()
    while line_n>"":
        st = line_n.strip('\n')
        line_title = fw.write(st+'\t')
        line_n = fn.readline()
    fw.write('\n')
    lines_a = fa.readline()
    i = 1
    while lines_a>'':
        print i
        line_list = lines_a.strip('\n').split('\001')
        name, mbl_num, id_card, loan_dt, label, value = line_list
        #iden_num, value = line_list
        fea_list, flag = v03.main(value, loan_dt=loan_dt)
        if flag == 0 and fea_list not in [[], '', None]:
            fea_list = [name, mbl_num, id_card, loan_dt, label] + fea_list
            #fea_list = [iden_num] + fea_list
            fea_str = '\t'.join(fea_list)
            fw.write(fea_str+'\n')
        i += 1
        lines_a = fa.readline()
