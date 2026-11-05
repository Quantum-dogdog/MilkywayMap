using UnityEngine;
using UnityEngine.SceneManagement;

public class CameraContro : MonoBehaviour
{
    public GameObject Target;
    public GameObject Target2;

    private float horiAngle;
    private float vertiAngle;

    public float distance;

    private Vector3 tem;

    public float mScrollValue = 10;

    private CharacterController cc;
    private float horizontalMove, verticalMove, shangxiaMove;
    public float moveSpeed;
    private Vector3 fangxiang;

    // ================= floating origin =================
    // Unity 的 Transform.position 是 Vector3(float)：坐标数值越大，相邻两个
    // 可表示的数之间隔得越远（|X|~2300 时约 2.4e-4，~50000 时约 3.9e-3）。
    // 相机每帧的小位移一旦小于这个间隔就被舍入掉，累积后一次跳一整格，
    // 即远处恒星抽搐。让相机始终待在原点附近即可消除：超过阈值时把整个
    // 场景（所有根物体）一起平移，物体间相对位置不变，画面不动。
    [Tooltip("相机离世界原点超过该距离时，把整个世界平移回原点附近")]
    public float originThreshold = 100f;

    // 累计平移量用 double 保存，便于以后还原绝对坐标 / 做 LOD 或物理
    private double originOffsetX = 0.0;
    private double originOffsetY = 0.0;
    private double originOffsetZ = 0.0;

    public void GetOriginOffset(out double x, out double y, out double z)
    {
        x = originOffsetX; y = originOffsetY; z = originOffsetZ;
    }

    void Start()
    {
        cc = GetComponent<CharacterController>();
        if (Target == null)
        {
            Target = GameObject.Find("Sun");
        }
        if (Target == null)
        {
            Debug.LogWarning("CameraContro: 未找到 Sun，也未在 Inspector 指定 Target。");
        }
        ApplyFloatingOrigin();
    }

    void Update()
    {
        cameraView();        // 按住左键旋转视角
        CameraZoom();        // 滚轮缩放
        cameraMove();        // Tab + 方向键自由移动
    }

    void LateUpdate()
    {
        MouseClick();

        if (Input.GetKey(KeyCode.Escape))
        {
            Application.Quit();
        }

        // 所有位置运算结束后再判断是否把世界拉回原点
        ApplyFloatingOrigin();
    }

    /// <summary>
    /// 相机远离原点时，把场景里所有根物体一起平移回原点附近。
    /// 只改根 Transform：子物体是局部坐标，会跟着父物体一起走，无需逐个处理。
    /// 相机本身也是根物体，平移后 transform.position 回到 0 附近，
    /// Target 与相机都变成小数值，大数相减、大数加小偏移的精度损失同时消失。
    /// </summary>
    private void ApplyFloatingOrigin()
    {
        Vector3 p = transform.position;
        if (p.magnitude < originThreshold) return;

        Vector3 shift = new Vector3(Mathf.Round(p.x), Mathf.Round(p.y), Mathf.Round(p.z));
        if (shift.sqrMagnitude < 1.0f) return;

        originOffsetX += shift.x;
        originOffsetY += shift.y;
        originOffsetZ += shift.z;

        foreach (GameObject root in SceneManager.GetActiveScene().GetRootGameObjects())
        {
            root.transform.position -= shift;
        }
    }

    private void cameraView()
    {
        if (!UnityEngine.Input.GetMouseButton(0)) return;
        if (Target == null) return;

        vertiAngle += UnityEngine.Input.GetAxis("Mouse Y");
        horiAngle -= UnityEngine.Input.GetAxis("Mouse X");

        Quaternion quaternion = Quaternion.Euler(vertiAngle, horiAngle, 0);
        transform.rotation = quaternion;

        distance = (Target.transform.position - transform.position).magnitude;
        tem = new Vector3(0, 0, -distance);
        Vector3 position = quaternion * tem + Target.transform.position;
        transform.position = position;
    }

    public void CameraZoom()
    {
        if (!UnityEngine.Input.GetMouseButton(0)) return;
        if (Camera.main == null) return;
        Vector3 dir = Camera.main.ScreenPointToRay(UnityEngine.Input.mousePosition).direction;
        transform.position += dir * UnityEngine.Input.GetAxis("Mouse ScrollWheel") * mScrollValue;
    }

    private void cameraMove()
    {
        if (!UnityEngine.Input.GetKey(KeyCode.Tab)) return;
        if (cc == null) return;

        horizontalMove = UnityEngine.Input.GetAxis("Horizontal") * moveSpeed;   // A / D 左右
        verticalMove = UnityEngine.Input.GetAxis("Vertical") * moveSpeed;       // W / S 前后
        shangxiaMove = UnityEngine.Input.GetAxis("Jump") * moveSpeed;           // 上下

        fangxiang = transform.forward * verticalMove + transform.right * horizontalMove + transform.up * shangxiaMove;
        cc.Move(fangxiang * Time.deltaTime);
    }

    void MouseClick()
    {
        if (Camera.main == null) return;
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        if (!Input.GetMouseButtonDown(1)) return;
        if (!Physics.Raycast(ray, out RaycastHit hitinfo)) return;

        // 直接用命中的对象。原写法先把 Target 置 null，再用
        // GameObject.Find(名字) 找回来：Find 可能返回 null（随后空引用），
        // 场景里有重名对象时还会选错。
        GameObject picked = hitinfo.transform.gameObject;
        if (picked == null) return;

        Target2 = Target;
        Target = picked;
        transform.LookAt(Target.transform.position);
    }
}
