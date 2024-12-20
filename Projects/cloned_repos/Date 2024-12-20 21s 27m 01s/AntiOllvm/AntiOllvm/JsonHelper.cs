using Newtonsoft.Json;

namespace AntiOllvm;

public class JsonHelper
{
    public static T Format<T>(string json)
    {
         
        return JsonConvert.DeserializeObject<T>(json);
    }

    public static string ToString(Object o)
    {
        return JsonConvert.SerializeObject(o);
    }
}