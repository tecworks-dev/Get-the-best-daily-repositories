import fs from 'fs';
import path from 'path';

// 创建一个空的示例结构
const emptyFamilyData = {
  "generations": [
    {
      "title": "示例世代",
      "people": [
        {
          "id": "example-person",
          "name": "示例姓名",
          "info": "示例信息"
        }
      ]
    }
  ]
};

// 确保config目录存在
const configDir = path.join(process.cwd(), 'config');
if (!fs.existsSync(configDir)) {
  fs.mkdirSync(configDir);
}

// 如果家族数据文件不存在，则创建一个示例文件
const familyDataPath = path.join(configDir, 'family-data.json');
if (!fs.existsSync(familyDataPath)) {
  fs.writeFileSync(
    familyDataPath, 
    JSON.stringify(emptyFamilyData, null, 2), 
    'utf8'
  );
  console.log(`创建了示例家族数据文件: ${familyDataPath}`);
} else {
  console.log(`家族数据文件已存在: ${familyDataPath}`);
}

// 提示用户下一步操作
console.log('\n配置说明:');
console.log('1. 编辑 config/family-data.json 文件添加您的家族数据');
console.log('2. 如需开启验证，在.env.local中设置 NEXT_PUBLIC_REQUIRE_AUTH=true');
console.log('3. 验证模式可在.env.local中设置，NEXT_PUBLIC_AUTH_MODE为"all"(所有家族成员) 或 "specific"(特定用户)');
console.log('4. 若使用specific模式，请在.env.local中设置 NEXT_PUBLIC_SPECIFIC_NAME 指定允许登录的用户名'); 