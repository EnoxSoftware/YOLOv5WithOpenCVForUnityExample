using UnityEngine;
using UnityEngine.SceneManagement;

namespace YOLOv5WithOpenCVForUnityExample
{

    public class ShowLicense : MonoBehaviour
    {

        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("YOLOv5WithOpenCVForUnityExample");
        }
    }
}
