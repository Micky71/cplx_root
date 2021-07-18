#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ipywidgets as widgets
from ipywidgets.widgets import Layout
import matplotlib.pyplot as plt
import math
import numpy as np
from IPython.display import display
import threading
import ctypes
import re
from time import sleep, time


# In[2]:


main = __name__ == "__main__"


# In[3]:


# constants

pi = math.pi
sin = math.sin
asin = math.sin
cos = math.cos
acos = math.acos
tan = math.tan
atan = math.atan
atan2 = math.atan2
rad = math.radians
deg = math.degrees
gamma = math.gamma
fact = math.factorial
floor = math.floor
ceil = math.ceil
log10 = math.log10
sqrt = math.sqrt
sign = np.sign
inf = np.inf


# In[4]:


get_ipython().run_cell_magic('html', '', '<style>\n    .bold_text{\n        color: black;\n        font-weight: bold;\n    }\n    .error_text{\n        color: red;\n        font-weight: bold;\n    }\n    .input_field{\n        border: 2px solid black;\n    }\n    .input_error{\n        border: 4px solid #F00;\n    }\n    .help{\n        color: #007;\n        background-color: #CFC;\n    }\n    .shadow{\n        box-shadow: 8px 8px 10px #444\n    }\n    .rational_widget{\n        background-color: #CCF;\n        border: #99F;\n    }\n    .complex_widget{\n        background-color: #FFA;\n        border: #FDA;\n    }\n</style>')


# In[5]:


def get_types_str(types):
    assert(type(types) is set), "types has got to be a set of types"
    if len(types) == 0:
        return "{}"
    types_str = "{"
    for t in types:
        assert(type(t) is type), "types has got to be a set of types"
        types_str += t.__name__ + ", "
    types_str = types_str[:-2] +"}"
    return types_str


# In[6]:


if main:
    print(get_types_str({int, float}))


# In[7]:


def precision(fp, prec=2, rnd=True, valid_types={int,float,complex}):
    if type(valid_types) is type:
        valid_types = {valid_types}
    fptype_str = get_types_str(valid_types)
    assert(type(fp) in valid_types and prec == int(prec)),        f"fp has got to be in {fptype_str}"
        
    def calc(fp):
        if fp == 0:
            return 0
        if fp >= 1 and rnd:
            return round(fp, prec)
        else:
            mg = int(floor(log10(abs(fp))))+1
            fact = 10 ** (prec-mg)
            return round(fp * fact)/fact
    
    
    if type(fp) is complex:
        return complex(calc(fp.real), calc(fp.imag))
    else:
        return calc(fp)


# In[8]:


if main:
    print(precision(123e-6,2))


# In[9]:


def greatest_common_factor(a, b):
    while b != 0:
        c = a % b
        a = b
        b = c
    return a


# In[10]:


if main:
    print(f"gcf(123,7):{greatest_common_factor(124,12):+g}")


# In[11]:


def text_to_html(text: str):
    text = text.strip()
    text = text.strip('\n')
    return text.replace('\n','<br>')


# In[12]:


def get_proper_frac(numtor, dentor, shorten=False, mult=1):

    assert(type(numtor) in {int, float}),        "numerator has got to be of class in {int, float}"
    
    assert(dentor == int(dentor) and dentor != 0),        "denuminator has got to be an none-zero int"
    
    assert(type(mult) in {int, float} and mult > 0),        "mult has got to be an int"

    whole = int(numtor/(mult*dentor))*mult
    
    numtor -= dentor*whole
        
    return whole, numtor, int(dentor)
        
if main:
    display(get_proper_frac(370.5,180, mult=2))


# In[13]:


def expression_to_value(expr_str, valid_types={int, float}):

    numtypes_str = get_types_str(valid_types)

    assert(type(expr_str) is str),        "expr_str has got to be of class str"

    valid_types_str = get_types_str(valid_types)

    expr_str = expr_str.replace(" ", "")
    
    if len(expr_str) == 0:
        number = int(0)
    else:
        pattern = r"([0-9\.\,]|[\*/\+\-//%\(\)]"
        pattern += r"|sqrt"
        pattern += r"|cos|acos|sin|asin|tan|atan|atan2"
        pattern += r"|floor|ceil|round"
        pattern += r"|e|pi|rad|deg|fact|gamma|complex)+"
        
        assert(re.fullmatch(pattern, expr_str) is not None),            "invalid expr_str"

        try:
            number = eval(expr_str)
        except Exception as ex:
            raise(ex)
            
    assert(type(number) in valid_types),        f"Not in valid_types={valid_types_str} but a {type(number).__name__}"
    
    return number

if main:
    display(expression_to_value("fact(4)+sin(rad(30))"))


# In[14]:


class Complex(object):
    
    def __init__(self, cplx_str, mod2pi=True):
        
        self.__isinitialized = False
        
        assert(type(mod2pi) is bool),            "mod has got to be of class bool"
        
        assert(type(cplx_str) is str),            "cplx_str has got to be of class str"
        
        defi_str = cplx_str = cplx_str.strip()
        
        parts = re.findall(r" \+i |\+!| \*i |\*!", cplx_str)

        assert(len(parts) < 2),            "cplx_str has got to conatain at most only 1 string in "            "{' +i ', '+!', ' *i ','*!'}"

        assert_str = ""
        if len(parts) == 0:
            try:
                re_part = expression_to_value(cplx_str)
            except Exception as ex:
                assert_str +=                    f"real part expression={cplx_str!r}:\n"                    f"{str(ex)}"
            else:
                im_part = 0
                angle = 0
                amount = abs(re_part)
            
        elif parts[0] in {' +i ','+!'}:
            re_expr, im_expr = cplx_str.split(parts[0])
            try:
                re_part = expression_to_value(re_expr)
            except Exception as ex:
                assert_str +=                    f"real part expression={re_expr!r}: "                    f"{str(ex)}\n"
            try:
                im_part = expression_to_value(im_expr)
            except Exception as ex:
                assert_str +=                    f"imag part expression={im_expr!r}: "                    f"{str(ex)}"
            if len(assert_str) == 0:
                angle = atan(im_part/re_part)
                amount = sqrt(re_part**2 + im_part**2)
            
            defi_str = re_expr + " +! " + im_expr
        else:
            amount_expr, angle_expr = cplx_str.split(parts[0])
            try:
                amount = expression_to_value(amount_expr)
            except Exception as ex:
                assert_str +=                    f"amount expression={amount_expr!r}: "                    f"{str(ex)}\n"
            else:
                if amount < 0:
                    assert_str +=                        f"amount = {amount} may not be negative"
            try:
                angle = expression_to_value(angle_expr)
            except Exception as ex:
                assert_str +=                    f"angle expression={angle_expr!r}: "                    f"{str(ex)}"
            if len(assert_str) == 0:
                re_part = amount * cos(angle)
                im_part = amount * sin(angle)

            defi_str = amount_expr + " *! " + angle_expr

        assert(len(assert_str) == 0), "\n" + assert_str.rstrip("\n")
        
        if mod2pi:
            angle %= 2*pi
            
        self.__defi_str = defi_str
        self.__mod = mod2pi
        self.__re_part = re_part
        self.__im_part = im_part
        self.__amount = amount
        self.__angle = angle

        self.__isinitialized = True
        
    def __repr__(self):
        if self.__isinitialized:
            return f"Complex({self.__defi_str!r}, mod={self.__mod}); "                f"{self.__re_part} + {self.__im_part}i; "                f"{self.__amount} * e^{self.__angle}i"
        else:
            return "Complex"
    
    @property
    def defi_str(self):
        return self.__defi_str
    @property
    def real(self):
        return self.__re_part
    
    @property
    def imag(self):
        return self.__im_part
    
    @property
    def amount(self):
        return self.__amount
    
    @property
    def angle(self):
        return self.__angle
    
    def get_latex(self, prec=3, deg_prec=3, plus=None):
        assert(type(prec) in {int, float} and prec == round(prec)),            "prec has got to have an interger value"
        
        assert((type(deg_prec) in {int, float} and deg_prec == round(deg_prec))
               or deg_prec is None),\
            "deg_prec has got to keep a integer value"
        
        assert(type(plus) in {int, float} or plus is None),            "plus has got to be in {int, float} or None"
        
        re_part = precision(self.__re_part, prec)
        im_part = precision(self.__im_part, prec)
        amount = precision(self.__amount, prec)
        angle = self.__angle
        parts_str = r"$" + f"{re_part:+g}  {im_part:+g}" + r"\,\rm\mathbf{i}" + r"$"
        
        if plus is None:
            if deg_prec is None:
                euler_str = r"$" + f"{amount} * " + r"\large{e}^{" + f"{precision(angle, prec)}"                    r"\rm{\,\mathbf i}}$"
            else:
                angle = deg(angle)
                whole, num, den = get_proper_frac(angle, 180, mult=2)
                num = precision(num, deg_prec)
                if num < 0:
                    sign = "-"
                else:
                    sign = ""
               
                euler_str = r"$" + f"{amount:g}" + r"\cdot e^{" + sign
                num = abs(num)
                if abs(whole) > 0:
                    
                    euler_str += r"\left(" + f"{abs(num):g}"                                 + r"{\large\frac{\pi}{180}}+"                                +f"{abs(whole):g}" + r"\pi\right)"
                else:
                    euler_str += f"{num:g}" + r"{\large\frac{\pi}{180}}"
                    
                euler_str += r"\rm\,{\mathbf{i}}}$"
        else:
            angle -= plus
            if deg_prec is None:
                euler_str = r"$" + f"{amount:g} " + r"\cdot e^{"                    f"{precision(plus, prec)}+{precision(angle, prec)}"                    r"\rm{\,\mathbf i}}$"
            else:
                angle = deg(angle)
                whole, num, den = get_proper_frac(angle, 180, mult=2)
                plus = precision(deg(plus), deg_prec)
                num = precision(num, deg_prec)
                
                euler_str = r"$" + f"{amount:g}" + r"\cdot e^{"

                if abs(whole) > 0:
                    
                    euler_str +=                        r"\left(\left(" + f"{plus:+g}{num:+g}" + r"\right)"                        r"{\large\frac{\pi}{180}}" + f"{whole:+}" + r"\pi\right)"
                else:
                    euler_str +=                        r"\left(" + f"{plus:+g}{num:+g}" + r"\right)"                        r"{\large\frac{\pi}{180}}"   

                euler_str += r"\rm\,{\mathbf{i}}}$"

            
        return parts_str, euler_str

    
if main:
    cplx_num = Complex("1/3 *i  rad(3*360)", mod2pi=False)
    w1 = widgets.HTMLMath(cplx_num.get_latex(prec=2, deg_prec=3)[0][:-1]
                          + r"\quad" + cplx_num.get_latex(prec=2, deg_prec=3)[1][1:])
    w2 = widgets.HTMLMath(cplx_num.get_latex(prec=2, deg_prec=3, plus=rad(7))[0][:-1]
                          + r"\quad" + cplx_num.get_latex(prec=2, deg_prec=3, plus=rad(7))[1][1:])
    display(widgets.VBox([w1,w2]))


# In[15]:


class ComplexWidget(object):
    
    mod_None_default = True
    observe_None_default = False
    
    def __init__(self, heading="Complex number", default_value="sqrt(2)*!rad(23)", 
                 style="full", mod2pi=None):

        self.__is_initialized = False
        
        assert(type(default_value) is str),            "default value has got to be of class str"
        
        assert(type(heading) is str),            "heading has got to be of class str"
        
        assert(type(style) is str and style in {'full', 'output', 'plain'}),            "style has got to be in {'full', 'plain'}"
        
        assert(type(mod2pi) is bool or mod2pi is None)
        
        self.__display_heading = self.__display_checkboxes            = self.__display_output = ""
        
        if len(heading) == 0:
            self.__display_heading = "none"
  
        if style == "output":
            self.__display_checkboxes = "none"
        elif style == "plain":
            self.__display_checkboxes = "none"
            self.__display_output = "none"
        
        if mod2pi is None:
            mod_init = ComplexWidget.mod_None_default
        else:
            mod_init = mod2pi
             
        self.__observers = []
        
        try:
            self.__complex = Complex(default_value, mod_init)
        except Exception as ex:
            assert(True is False), str(ex)
    
        self.__default_value = default_value
        self.__heading = heading
        self.__style = style
        self.__mod = mod2pi
        self.__is_initialized = True
        
    def __create_widget(self, VBox_width):
        
        self.__help = False
        
        def evaluate(sender):
            dbg.value = "evaluate" + f"complex_input.value"
            try:
                self.__complex = Complex(complex_input.value, mod_checkbox.value)
            except Exception as ex:
                dbg.value += " exception"
                complex_output.value = text_to_html(str(ex))
                complex_output.add_class('error_text')
                complex_output.layout.display = ""
                complex_input.add_class('error_input')
                self.__complex = None
            else:
                htmlmath_str = self.__complex.get_latex()[0][:-1]                    + r",\quad" + self.__complex.get_latex()[1][1:]
                complex_output.value = htmlmath_str
                complex_output.layout.display = self.__display_output
                complex_output.remove_class('error_text')
                complex_input.remove_class('error_input')
                dbg.value += " " + htmlmath_str
            finally:
                if sender != "put_number":
                    self.__notify()
                    
        self.__evaluate = evaluate
        
        def on_click_help(button):
            self.__help = not self.__help
            if self.__help:
                help_text.layout.display = ''
            else:
                help_text.layout.display = 'none'

        def on_toggle_mod(change):
            evaluate("dummy")
            
        def observe_input(change):
            evaluate('dummy')
            
        def on_toggle_observe(change):
            observe = change['new']
            if observe:
                complex_input.observe(observe_input, names='value')
                self.__evaluate("dummy")
            else:
                complex_input.unobserve(observe_input, names="value")
                        
        #########################################################################
        
        heading = widgets.HTMLMath(self.__heading,
                                   layout=Layout(width="auto"))
        heading.layout.display = self.__display_heading
        heading.add_class("bold_text")

        if self.__mod is None:
            disabled = False
            value = ComplexWidget.mod_None_default
        else:
            disabled = True
            value = self.__mod
            
        mod_checkbox = widgets.Checkbox(value=value,
                                        disabled=disabled,
                                        description="mod2pi",
                                        indent=False)
        mod_checkbox.observe(on_toggle_mod, names='value')
        
        observe_checkbox = widgets.Checkbox(
            value=False,
            description='observe',
            disabled=False,
            indent=False)
        observe_checkbox.observe(on_toggle_observe, names='value')
        
        checkboxes = widgets.HBox([mod_checkbox, observe_checkbox])
        checkboxes.layout.display = self.__display_checkboxes
        
        complex_input = widgets.Text(
                value=self.__default_value,
                layout=Layout(width="100%"),
                placeholder='',
                description='',
                disabled=False
            )
        complex_input.add_class("input_field")
        complex_input.on_submit(evaluate)
        
        help_button = widgets.Button(description='?',
                                     layout=Layout(width="2em"))
        help_button.add_class('help')
        help_button.on_click(on_click_help)
        
        help_text = widgets.HTMLMath(
            layout=Layout(width="auto"),
            value="<b>Complex</b>Widget: In the textfield "\
                  "you may put a text like <b>'0+!1'</b> or <b>'1*!cos(deg(90))'</b>:<br>"\
                  r"The 're_part +! im_part' indicates $re\_part + im\_part\, \rm\mathbf{i}$<br>"
                  r"The 'amount *! angle' indicates $amount \cdot e^{angle\,\rm\mathbf{i}}$")
        help_text.add_class('help')
        help_text.layout.display = 'none'
              
        complex_output = widgets.HTMLMath(layout=Layout(width="auto"))
    
        ###############################################################
        dbg = widgets.HTML(value="debug")
        dbg.layout.display = "none"
        ###############################################################
        
        input_help = widgets.HBox([complex_input, help_button],
                                  layout=Layout(width="auto"))
        
        wdgt = widgets.VBox([heading, checkboxes, input_help,
                             help_text, complex_output, dbg],
                            layout=Layout(width=VBox_width))

        wdgt.add_class('complex_widget')
        evaluate("put_number")
        return wdgt
       
    def observe(self, observer):
        assert(callable(observer)),            "observer has got to be callable"
        assert(observer not in self.__observers),            f"{observer.__name__!r} already in observer list"
        self.__observers.append(observer)
    
    def unobserve(self, observer):
        assert(observer in self.__observers),            f"{observer} not in observer list"
        self.__observers.remove(observer)
        
    def __notify(self):
        for observer in self.__observers:
            observer(self.__complex)
        
    def get_widget(self, VBox_width="30%"):
        self.__widgets = self.__create_widget(VBox_width)
        return self.__widgets
    
    @property
    def complex_number(self):
        self.__evaluate('put_number')
        return self.__complex
        
    def __repr__(self):
        if self.__is_initialized:
            return f"ComplexWidget(self, {self.__default_value!r}, "                       f"{self.__heading!r}, {self.__style!r})"
        else:
            return "ComplexWidget: init error"


# In[16]:


if main:
    display(ComplexWidget().get_widget())


# In[17]:


class Rational(object):
    """Convert string into rational number. Possible strings:
        0.33, +.33, -.33, 1/3 = 0.p3 = .p3 (p follow periodic numbers)"""

    def __init__(self, ratio_str,
                 num_interval=[-inf, inf],
                 den_interval=[-inf, inf],
                 shorten=True,):
        
        
        self.__is_intitialized = False
        
        assert(type(ratio_str) is str),            "ratio_str has got to be of class str"
        assert(type(shorten) is bool),            "shorten has to be of class bool"

        assert(np.all(np.array([num_interval, den_interval])
                   == np.round(np.array([num_interval, den_interval])))
               and num_interval == sorted(num_interval)
               and den_interval == sorted(den_interval)),\
            "num_interval and den_interval have to be sorted integers"
               
        self.__defi_str = ratio_str.strip()
        self.__num_interval = num_interval
        self.__den_interval = den_interval
        self.__shorten = shorten
        self.__periodic = False
        self.__defi_str, self.__numerator, self.__denominator            = self.__evaluate()
        self.__is_intitialized = True

    def __get_decfrac(self, dec_str):
        dec_str = dec_str.strip().lower()
        sign = 1
        signsign = ""
        match = re.match('[+|-]', dec_str)
        if match is not None:
            if match[0][0] == "-":
                sign = -1
                signsign = "-"
            dec_str = dec_str[1:]

        numerator = denominator = None
        
        if re.fullmatch(r"[0-9]+", dec_str) is not None:  # '12', '23'
            numerator = int(dec_str)
            denominator = 1

        elif re.fullmatch(r"([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)", dec_str)                is not None:    # # '.2', '1.', '2.3'
            nat, decs = dec_str.split('.')

            if len(nat) == 0:
                nat = 0
            else:
                nat = int(nat)

            denominator = 10 ** len(decs)
            if denominator > 1:
                numerator = nat * denominator + int(decs)
            else:
                numerator = nat * denominator

        elif re.fullmatch(r'[0-9]*\.[0-9]*p[0-9]+', dec_str)                is not None:  # # '0.p3', '.0p34'
            nat_str, dec_str_, peri_str = re.split(r"\.|p", dec_str)
            full_mult = 10 ** len(dec_str_+peri_str)
            dec_mult = 10 ** len(dec_str_)
            numerator = int(nat_str+dec_str_+peri_str) - int("0"+nat_str+dec_str_)
            denominator = full_mult-dec_mult
            self.__periodic = True

        if numerator is not None:
            numerator *= sign
        return signsign + dec_str, numerator, denominator

    def __evaluate(self):
        ratio_str = self.__defi_str
        fracs = re.findall("/", ratio_str)
        
        if len(fracs) == 0:
            ratio_str, numerator, denominator = self.__get_decfrac(ratio_str)
            assert(numerator is not None),                f"{ratio_str!r} doesn't match any pattern"
        elif len(fracs) == 1:
            num_str, den_str = ratio_str.split("/")
            
            num_str, num_numtor, num_dentor = self.__get_decfrac(num_str)
            den_str, den_numtor, den_dentor = self.__get_decfrac(den_str)
            assert(None not in {num_numtor, den_numtor}),                f"{ratio_str!r} doesn't match any pattern"

            numerator = num_numtor * den_dentor
            denominator = den_numtor * num_dentor
            
            ratio_str = num_str + " / " + den_str
        else:
            assert(True is False),                f"{ratio_str!r} has to contain at most one '/' sign"
        
        assert(denominator != 0),            f"{ratio_str!r} denominator may not be zero"

        if self.__shorten is True:
            gcf = greatest_common_factor(numerator, denominator)
            numerator /= gcf
            denominator /= gcf
            
        assert(numerator >= self.__num_interval[0]
               and numerator <= self.__num_interval[1]),\
            f"{ratio_str!r} numerator has got to be in {self.__num_interval}"\
            f" but is {numerator:g}"
        
        assert(denominator >= self.__den_interval[0]
               and denominator <= self.__den_interval[1]),\
            f"{ratio_str!r} denominator has got to be in {self.__den_interval},"\
            f" but is {denominator:g}"
            
        return ratio_str, int(numerator), int(denominator)
    
    def shorten(self):
        self.__shorten = True
        ratio_str, self.__numerator, self.__denominator            = self.__evaluate()
        
    def unshorten(self):
        self.__shorten = False
        ratio_str, self.__numerator, self.__denominator            = self.__evaluate()
        
    @property
    def defi_str(self):
        return self.__defi_str
        
    @property
    def fraction(self):
        return self.__numerator, self.__denominator
    
    @property
    def numerator(self):
        return self.__numerator

    @property
    def denominator(self):
        return self.__denominator
    
    def get_latex(self, prec=3, peri_prec=12):

        assert(prec == int(prec)),            "float_prec has got to be an int"
        assert(peri_prec == int(peri_prec)),            "peri_prec has got to be an int"
        
        if self.__periodic:
            prec = peri_prec
        
        float_str = f"{self.__numerator/self.__denominator:0.{prec}g}"
                 
        frac_str = r"${\large\frac{" + f"{self.__numerator}"            r"}{" + f"{self.__denominator}" + r"}}$"
        
        float_str = r"$" + float_str + "$"
        
        return frac_str, float_str
    
    def get_widgets_text(self):
        return f"{self.__numerator}/{self.__denominator} = {self.__numerator/self.__denominator}"
        
    def __repr__(self):
        if self.__is_intitialized:
            
            return (f"Rational({self.__defi_str!r}, num_interval={self.__num_interval}, "                    f"den_interval={self.__den_interval}, shorten={self.__shorten}); "                    f"{(self.__numerator)}" + "/" + f"{(self.__denominator)}; "                    + str(format(self.__numerator/self.__denominator)))
        else:
            return "Rational: init - error"


# In[18]:


if main:
    rat_num = Rational("7.42p0023")
    display(widgets.HTMLMath(rat_num.get_latex()[0][:-1] + r"\quad" + rat_num.get_latex()[1][1:]))


# In[19]:


class RationalWidget(object):
    shorten_None_default = True
    observe_None_default = False
    
    def __init__(self, heading="Rational number", default_value=".p23",
                 num_interval=[-inf, inf],
                 den_interval=[-inf, inf],
                 shorten=None,
                 observe=None,
                 style="full"):

        self.__is_initialized = False

        assert(type(shorten) is bool or shorten is None),            "shorten has got to be of class bool or None"
        
        assert(type(observe) is bool or observe is None),            "observe has got to be of class bool or None"
        
        assert(type(heading) is str),            "heading has got to be of class str"
        
        assert(type(style) is str and style in {'plain', 'output', 'full'}),            "style has got to be in {'plain', 'output', 'full'}"

        self.__display_heading = self.__display_checkboxes            = self.__display_output = ""
        
        if len(heading) == 0:
            self.__display_heading = "none"

        if style == 'plain':
            self.__display_checkboxes = "none"
            self.__display_output = "none"
        else:
            self.__display_output = ""
            
        if style == 'output':
            self.__display_checkboxes = "none"

        if shorten is None:
            shorten_init = RationalWidget.shorten_None_default
        else:
            shorten_init = shorten
        try:
            self.__rational = Rational(default_value,
                                       num_interval,
                                       den_interval,
                                       shorten_init,)
        except Exception as ex:
            assert(True is False),                "RationalWidget: " + str(ex)
            
        self.__shorten = shorten
        self.__num_interval = num_interval
        self.__den_interval = den_interval
        self.__observe = observe
        self.__heading = heading
        self.__style = style
        self.__default_value = default_value
        self.__observers = []
        self.__help = False
        self.__is_initialized = True
        
    def __create_widget(self, VBox_width):
        
        def evaluate(sender):
            dbg.value="evaluate"
            try:
                self.__rational = Rational(rational_input.value,
                                           self.__num_interval,
                                           self.__den_interval,
                                           shorten_checkbox.value,)
            except Exception as ex:
                dbg.value += " exception"
                rational_output.value = str(ex)
                rational_output.add_class('error_text')
                rational_input.add_class('error_input')
                rational_output.layout.display = ""
                self.__rational = None
            else:
                htmlmath_str = self.__rational.get_latex()[0][:-1]                    + ",\quad" + self.__rational.get_latex()[1][1:]
                rational_output.value = htmlmath_str
                rational_output.remove_class('error_text')
                rational_output.layout.display = self.__display_output
                rational_input.remove_class('error_input')

                dbg.value += " " + htmlmath_str
            finally:
                if sender != "put_number":
                    self.__notify()
                    
        self.__evaluate = evaluate
        
        def observe_input(change):
            evaluate('dummy')
            
        def on_toggle_checkboxes(change):
            evaluate("dummy")
            
        def on_toggle_observe(change):
            observe = change['new']
            if observe:
                rational_input.observe(observe_input, names='value')
                evaluate('dummy')
            else:
                rational_input.unobserve(observe_input, names='value')
                
        def on_click_help(button):
            self.__help = not self.__help
            if self.__help:
                help_text.layout.display=''
            else:
                help_text.layout.display='none'
                
#########################################################################
                
        if self.__shorten is None:
            disabled = False
            value = RationalWidget.shorten_None_default
        else:
            disabled = True
            value = self.__shorten
        shorten_checkbox = widgets.Checkbox(
            value=value,
            description='shorten',
            disabled=disabled,
            indent=False)
        shorten_checkbox.observe(on_toggle_checkboxes, names='value')
        
        if self.__observe is None:
            disabled = False
            value = RationalWidget.observe_None_default
        else:
            disabled = True
            value = self.__observed
        observe_checkbox = widgets.Checkbox(
            value=value,
            description='observe',
            disabled=disabled,
            indent=False)
        observe_checkbox.observe(on_toggle_observe, names='value')

        heading = widgets.HTMLMath(self.__heading, layout=Layout(width="auto"))
        heading.add_class('bold_text')
        heading.layout.display = self.__display_heading

        rational_input = widgets.Text(
                value=self.__default_value,
                layout=Layout(width="100%"),
                placeholder='',
                description='',
                disabled=False
            )
        rational_input.add_class("input_field")
        if self.__observe is True:
            rational_input.observe(observe_input)
        rational_input.on_submit(evaluate)
        
        help_button = widgets.Button(description='?', layout=Layout(width="2em"))
        help_button.add_class('help')
        help_button.on_click(on_click_help)
        
        help_text = widgets.HTMLMath(layout=Layout(width="auto"),
            value="<b>Rational</b>Widget: In the textfield you may put a text like <b>'0.p3'</b> or <b>'-1/3'</b>.<br>"
                    + "The p has to be typed before the periodic numbers:<br>"
                    + r"$\mathbf{12.0p25 = 12.0\overline{25} = 12.02525252525\, . . .}$")
        help_text.add_class('help')
        help_text.layout.display = 'none'

        rational_output = widgets.HTMLMath(layout=Layout(width="auto"))
        rational_output.layout.display = self.__display_output
        
        input_row = widgets.HBox([rational_input, help_button])
        checkboxes = widgets.HBox([shorten_checkbox, observe_checkbox])
        
        checkboxes.layout.display = self.__display_checkboxes
        
        ###############################################################
        dbg = widgets.HTML(value="debug")
        dbg.layout.display = "none"
        ###############################################################
        
        wdgt = widgets.VBox([heading, checkboxes, input_row, help_text,
                             rational_output, dbg], layout=Layout(width=VBox_width))
        wdgt.add_class('rational_widget')
        evaluate("put_number")
        return wdgt
       
    def observe(self, observer):
        assert(callable(observer)),            'observer has got to be callable'
        assert(observer not in self.__observers),            f"{observer.__name__!r} already in observer list"
        self.__observers.append(observer)
    
    def unobserve(self, observer):
        assert(observer in self.__observers),            f"{observer.__name__} not in observer list"
        self.__observers.remove(observer)
        
    def __notify(self):
        for observer in self.__observers:
            observer(self.__rational)
        
    def get_widget(self, VBox_width="30%"):
        self.__widget = self.__create_widget(VBox_width)
        return self.__widget
    
    @property
    def rational_number(self):
        self.__evaluate('put_number')
        return self.__rational
        
    def __repr__(self):
        if self.__is_initialized:
            return  f"RationalWidget(self, heading={self.__heading!r}"                    f"default_value={self.__default_value!r}, "                    f"num_interval={self.__num_interval}, "                    f"den_interval={self.__den_interval}, "                    f"shorten={self.__shorten}, "                    f"observe={self.__observe}, "                    f"style={self.__style!r})"
        else:
            return "RationalWidget, init error"


# In[20]:


if main:
    
    wdgt1 = RationalWidget(heading=r"$Rational\,number$", num_interval=[1, inf],
                           den_interval=[2, inf], style='full')
    wdgt2 = ComplexWidget(heading="$\int_\Omega x\; d\Omega$", style='full')
    output = widgets.HTMLMath(layout = Layout(width="30%"))
    wdgt = widgets.VBox([wdgt1.get_widget(), wdgt2.get_widget(),output])
    display(wdgt)
    output.value += "<br>"
    def observer(number):
        rat_num = wdgt1.rational_number
        cplx_num = wdgt2.complex_number
        if None in [rat_num, cplx_num]:
            output.value += "Invalid fields<br>"
        else:
            output.value += rat_num.get_latex()[0] + "<br>"
            output.value += cplx_num.get_latex()[1] + "<br>"
            
    wdgt1.observe(observer)
    wdgt2.observe(observer)


# In[21]:


class Interruptable_thread(threading.Thread):
    
    def __init__(self, fcn, *args, **kwargs):
        threading.Thread.__init__(self)
        self.__fcn = fcn
        self.__args = args
        self.__kwargs = kwargs
              
    def run(self):
  
        try:
            self.__fcn(*self.__args, **self.__kwargs)
        finally:
            pass
        
    def get_id(self):
  
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def interrupt(self):  # raise_exception
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')


# In[22]:


if main:
    def testFcn(x=1):
        counter = 0
        while counter < 20:
            counter += 1
            sleep(.1)
            print(counter*x)

        
    t1 = Interruptable_thread(testFcn,1)
    t1.start()
    sleep(1)
    t1.interrupt()
    t1.join()
    print("End")
    
    t1 = Interruptable_thread(testFcn,1)
    t1.start()
    sleep(1)
    t1.interrupt()
    t1.join()
    print("End")


# In[23]:


def colorstr_to_floats(color):
    assert(type(color) is str), "color has got to be of class str"
    assert(re.fullmatch("#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})", color) is not None),        f"{color!r}: not a valid color-string"
    
    def hexbyte_to_int(hex_str):
        val = 0
        if hex_str[0] in "ABCDEF":
            val += 16*(ord(hex_str[0])-55)
        else:
            val += 16 * int(hex_str[0])

        if hex_str[1] in "ABCDEF":
            val += ord(hex_str[1])-55
        else:
            val += int(hex_str[1])
        return val
    
    if len(color) == 4:
        color = "#" + 2*color[1] + 2*color[2] + 2*color[3]
    
    color = color.upper()
    floats = []
    for i in range(3):
        floats  += [hexbyte_to_int(color[1+i*2:3+i*2])/255]
    
    return floats


# In[24]:


def floats_to_colorstr(floats):
    def float_to_hexbytestr(val):
        val = round(val*255)
        retval = hex(val)[2:]
        if len(retval) == 1:
            retval += "0"
        return retval
    
    assert(type(floats) is list and len(floats) == 3),        "floats has got to be a 3-element list"
    assert(np.min(floats) >= 0 and np.max(floats) <= 1),        "floats has got to contain floats in [0,1]"
    
    colorstr = "#"
    for i in floats:
        colorstr += float_to_hexbytestr(i)
        
    return colorstr


# In[25]:


def brighten_color(color, brightness):
    assert(type(brightness) in {float, int} and brightness >= 0
           and brightness <= 1),\
        "bright has got to be of class float in [0, 1]"

    


# In[26]:


if main:
    print(floats_to_colorstr(colorstr_to_floats("#00a005")))


# In[29]:


if main:
    get_ipython().system('jupyter nbconvert minumbers.ipynb --to script')
else:
    print("numbers imported")


# In[ ]:





# In[ ]:




