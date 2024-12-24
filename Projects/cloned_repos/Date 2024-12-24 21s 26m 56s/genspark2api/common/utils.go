package common

import (
	"encoding/base64"
	"fmt"
	"github.com/google/uuid"
	jsoniter "github.com/json-iterator/go"
	_ "github.com/pkoukk/tiktoken-go"
	"math/rand"
	"strings"
	"time"
	"unicode/utf8"
)

// splitStringByBytes 将字符串按照指定的字节数进行切割
func SplitStringByBytes(s string, size int) []string {
	var result []string

	for len(s) > 0 {
		// 初始切割点
		l := size
		if l > len(s) {
			l = len(s)
		}

		// 确保不在字符中间切割
		for l > 0 && !utf8.ValidString(s[:l]) {
			l--
		}

		// 如果 l 减到 0，说明 size 太小，无法容纳一个完整的字符
		if l == 0 {
			l = len(s)
			for l > 0 && !utf8.ValidString(s[:l]) {
				l--
			}
		}

		result = append(result, s[:l])
		s = s[l:]
	}

	return result
}

func Obj2Bytes(obj interface{}) ([]byte, error) {
	// 创建一个jsonIter的Encoder
	configCompatibleWithStandardLibrary := jsoniter.ConfigCompatibleWithStandardLibrary
	// 将结构体转换为JSON文本并保持顺序
	bytes, err := configCompatibleWithStandardLibrary.Marshal(obj)
	if err != nil {
		return nil, err
	}
	return bytes, nil
}

func GetUUID() string {
	code := uuid.New().String()
	code = strings.Replace(code, "-", "", -1)
	return code
}

// RandomElement 返回给定切片中的随机元素
func RandomElement[T any](slice []T) (T, error) {
	if len(slice) == 0 {
		var zero T
		return zero, fmt.Errorf("empty slice")
	}

	// 确保每次随机都不一样
	rand.Seed(time.Now().UnixNano())

	// 随机选择一个索引
	index := rand.Intn(len(slice))
	return slice[index], nil
}

func SliceContains(slice []string, str string) bool {
	for _, item := range slice {
		if strings.Contains(str, item) {
			return true
		}
	}
	return false
}

func IsImageBase64(s string) bool {
	// 检查字符串是否符合数据URL的格式
	if !strings.HasPrefix(s, "data:image/") || !strings.Contains(s, ";base64,") {
		return false
	}

	if !strings.Contains(s, ";base64,") {
		return false
	}

	// 获取";base64,"后的Base64编码部分
	dataParts := strings.Split(s, ";base64,")
	if len(dataParts) != 2 {
		return false
	}
	base64Data := dataParts[1]

	// 尝试Base64解码
	_, err := base64.StdEncoding.DecodeString(base64Data)
	return err == nil
}

func IsBase64(s string) bool {
	// 检查字符串是否符合数据URL的格式
	//if !strings.HasPrefix(s, "data:image/") || !strings.Contains(s, ";base64,") {
	//	return false
	//}

	if !strings.Contains(s, ";base64,") {
		return false
	}

	// 获取";base64,"后的Base64编码部分
	dataParts := strings.Split(s, ";base64,")
	if len(dataParts) != 2 {
		return false
	}
	base64Data := dataParts[1]

	// 尝试Base64解码
	_, err := base64.StdEncoding.DecodeString(base64Data)
	return err == nil
}
