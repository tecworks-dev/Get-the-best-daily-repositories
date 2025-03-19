using UnityEngine;
using Newtonsoft.Json.Linq;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering;

namespace MCPServer.Editor.Commands
{
    /// <summary>
    /// Handles material-related commands
    /// </summary>
    public static class MaterialCommandHandler
    {
        /// <summary>
        /// Sets or modifies a material on an object
        /// </summary>
        public static object SetMaterial(JObject @params)
        {
            string objectName = (string)@params["object_name"] ?? throw new System.Exception("Parameter 'object_name' is required.");
            var obj = GameObject.Find(objectName) ?? throw new System.Exception($"Object '{objectName}' not found.");
            var renderer = obj.GetComponent<Renderer>() ?? throw new System.Exception($"Object '{objectName}' has no renderer.");

            // Check if URP is being used
            bool isURP = GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset;

            // Create material with appropriate shader based on render pipeline
            Material material;
            if (isURP)
            {
                material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            }
            else
            {
                material = new Material(Shader.Find("Standard"));
            }

            if (@params.ContainsKey("material_name")) material.name = (string)@params["material_name"];
            if (@params.ContainsKey("color"))
            {
                var colorArray = (JArray)@params["color"] ?? throw new System.Exception("Invalid color parameter.");
                if (colorArray.Count != 3) throw new System.Exception("Color must be an array of 3 floats [r, g, b].");
                material.color = new Color((float)colorArray[0], (float)colorArray[1], (float)colorArray[2]);
            }
            renderer.material = material;
            return new { material_name = material.name };
        }
    }
}