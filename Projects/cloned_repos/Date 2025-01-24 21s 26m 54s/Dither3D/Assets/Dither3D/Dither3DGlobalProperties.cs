using System.Collections.Generic;
using UnityEngine;

public class OverridePropertyAttribute : PropertyAttribute { }

[ExecuteAlways]
public class Dither3DGlobalProperties : MonoBehaviour
{
    static List<Material> ditherMaterials = new List<Material>();

    [Header("Global Options")]
    public bool radialCompensation;
    public bool quantizeLayers;
    public bool debugFractal;

    [Header("Global Overrides")]
    public bool saveInMaterials;

    [Space]
    
    [OverrideProperty] public float inputExposure = 1;
    [HideInInspector] public bool inputExposureOverride;
    
    [OverrideProperty] public float inputOffset = 0;
    [HideInInspector] public bool inputOffsetOverride;

    [Space]

    [OverrideProperty] public float dotScale = 5;
    [HideInInspector] public bool dotScaleOverride;
    
    [OverrideProperty] public float dotSizeVariability = 0;
    [HideInInspector] public bool dotSizeVariabilityOverride;
    
    [OverrideProperty] public float dotContrast = 1;
    [HideInInspector] public bool dotContrastOverride;
    
    [OverrideProperty] public float stretchSmoothness = 1;
    [HideInInspector] public bool stretchSmoothnessOverride;

    void OnEnable()
    {
        CollectMaterials();
        UpdateProperties();
    }

    void OnDisable()
    {
    }

    void OnValidate()
    {
        if (ditherMaterials != null && ditherMaterials.Count > 0 && ditherMaterials[0] == null)
            CollectMaterials();

        UpdateProperties();
    }

    void CollectMaterials()
    {
        ditherMaterials.Clear();
        Material[] materials = Resources.FindObjectsOfTypeAll<Material>();
        foreach (var mat in materials)
        {
            if (mat.HasProperty("_DitherTex"))
            {
                ditherMaterials.Add(mat);
            }
        }
    }

    void UpdateProperties()
    {
        bool changed = false;
        if (inputExposureOverride)
            SetShaderOverride("_InputExposure", inputExposure, ref changed);
        if (inputOffsetOverride)
            SetShaderOverride("_InputOffset", inputOffset, ref changed);
        if (dotScaleOverride)
            SetShaderOverride("_Scale", dotScale, ref changed);
        if (dotSizeVariabilityOverride)
            SetShaderOverride("_SizeVariability", dotSizeVariability, ref changed);
        if (dotContrastOverride)
            SetShaderOverride("_Contrast", dotContrast, ref changed);
        if (stretchSmoothnessOverride)
            SetShaderOverride("_StretchSmoothness", stretchSmoothness, ref changed);

        #if UNITY_EDITOR
        if (changed && saveInMaterials)
        {
            foreach (var mat in ditherMaterials)
            {
                UnityEditor.EditorUtility.SetDirty(mat);
            }
        }
        #endif

        EnableKeyword("RADIAL_COMPENSATION", radialCompensation);
        EnableKeyword("QUANTIZE_LAYERS", quantizeLayers);
        EnableKeyword("DEBUG_FRACTAL", debugFractal);
    }

    void EnableKeyword(string keyword, bool enable)
    {
        if (enable)
            Shader.EnableKeyword(keyword);
        else
            Shader.DisableKeyword(keyword);
    }

    void SetShaderOverride(string property, float value, ref bool changed)
    {
        foreach (var mat in ditherMaterials)
        {
            if (mat.GetFloat(property) != value)
            {
                mat.SetFloat(property, value);
                changed = true;
            }
        }
    }
}
