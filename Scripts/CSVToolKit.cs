using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public static class CSVToolKit
{
   public static void LoadFile(string path, Action<List<string[]>> a)
    {
        if (!File.Exists(path))
        {
            Debug.LogError(path + "no found");
            return;

         }
        StreamReader sr = null;
        try
        {
            sr = File.OpenText(path);
            List<string[]> content = new List<string[]>();
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                content.Add(line.Split(','));     //csv

             }

            sr.Close();
            sr.Dispose();
            a?.Invoke(content);

        }
        catch (Exception ex)
        {
            Debug.LogError(ex.Message);
        }

    }
}
